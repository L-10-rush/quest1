"""Streamlit web UI for video-dialogue-finder.

A thin presentation layer over the exact same `DialoguePipeline` the CLI
(`src/main.py`) uses -- this module imports `build_pipeline` and
`append_session_log` from there instead of wiring stage implementations
itself, so `main.py` stays the one composition root that decides which
concrete downloader/transcriber/matcher/etc. get used (see its own
docstring). This file only turns `PipelineConfig` + form input into calls
against that pipeline and renders the result.

Run with:
    uv run --group web streamlit run src/webapp/app.py

(`uv sync --group web` first, to install Streamlit -- it's optional and
deliberately not part of the default install; see pyproject.toml.)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit executes this file directly (not via `python -m`), so the repo
# root isn't automatically on sys.path the way it is for `python -m src.main`.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st

from src.config import (
    DEFAULT_ENGINE,
    DEFAULT_LANGUAGE,
    DEFAULT_MATCH_THRESHOLD,
    DEFAULT_WHISPER_MODEL,
    PipelineConfig,
)
from src.exceptions import PipelineError
from src.main import append_session_log, build_pipeline
from src.pipeline import PipelineRunSummary

st.set_page_config(page_title="video-dialogue-finder", page_icon="🎬", layout="wide")

STATE_KEYS = ("pipeline", "session", "config", "history")


def _reset_session() -> None:
    pipeline = st.session_state.get("pipeline")
    session = st.session_state.get("session")
    if pipeline is not None and session is not None:
        pipeline.cleanup(session)
    for key in STATE_KEYS:
        st.session_state.pop(key, None)


def _render_result(target_text: str, summary: PipelineRunSummary, log_path: Path | None) -> None:
    left, right = st.columns([3, 2])

    with left:
        if summary.best_frame_image is not None:
            st.image(summary.best_frame_image, channels="BGR", width="stretch")

    with right:
        st.metric("Timestamp", summary.timestamp)
        c1, c2 = st.columns(2)
        c1.metric("Frame", summary.frame_number)
        c2.metric("Match score", f"{summary.match_score:.1f}")
        st.write(f'**Matched text:** "{summary.matched_text}"')

        if summary.is_uncertain:
            st.warning(f"Result flagged UNCERTAIN -- {summary.uncertainty_reason}")

        if summary.screen_status is not None:
            badge = {"on_screen": st.success, "off_screen": st.error, "uncertain": st.warning}
            badge.get(summary.screen_status, st.info)(
                f"On-screen: **{summary.screen_status.upper()}** "
                f"(confidence {summary.screen_confidence:.2f}) -- {summary.screen_reason}"
            )

        st.caption(f"Elapsed {summary.total_seconds:.1f}s")

        frame_path = Path(summary.frame_image_path)
        json_path = Path(summary.result_json_path)
        dc1, dc2 = st.columns(2)
        if frame_path.exists():
            dc1.download_button(
                "Download frame PNG",
                frame_path.read_bytes(),
                file_name=frame_path.name,
                mime="image/png",
                key=f"png-{json_path.stem}",
            )
        if json_path.exists():
            dc2.download_button(
                "Download result JSON",
                json_path.read_bytes(),
                file_name=json_path.name,
                mime="application/json",
                key=f"json-{json_path.stem}",
            )
        if log_path is not None:
            st.caption(f"Logged to `{log_path}`")

    if summary.candidate_previews:
        with st.expander(f"Other candidates ({len(summary.candidate_previews)}, for ambiguity review)"):
            cols = st.columns(len(summary.candidate_previews))
            for col, (text, score, image) in zip(cols, summary.candidate_previews, strict=True):
                with col:
                    st.image(image, channels="BGR", width="stretch")
                    st.caption(f'score={score:.1f}  "{text}"')


st.title("🎬 video-dialogue-finder")
st.caption(
    "Find the exact frame where a spoken line of dialogue occurs -- a "
    "Streamlit front end over the same pipeline the CLI (`python -m src.main`) uses."
)

with st.sidebar:
    st.header("1. Load a video")
    prepared = "session" in st.session_state

    with st.form("session_form", border=True):
        video_url = st.text_input(
            "Video URL",
            value="" if not prepared else st.session_state["config"].video_url,
            placeholder="https://<URL>",
            disabled=prepared,
        )
        with st.expander("Advanced options", expanded=False):
            language = st.text_input("Language (ISO-639-1)", value=DEFAULT_LANGUAGE, disabled=prepared)
            engine = st.selectbox(
                "Transcription engine", ["whisperx", "vosk"],
                index=["whisperx", "vosk"].index(DEFAULT_ENGINE), disabled=prepared,
            )
            whisper_model = st.selectbox(
                "WhisperX model size",
                ["tiny", "base", "small", "medium", "large-v3"],
                index=["tiny", "base", "small", "medium", "large-v3"].index(DEFAULT_WHISPER_MODEL),
                disabled=prepared or engine != "whisperx",
            )
            device = st.selectbox("Device", ["cpu", "cuda"], disabled=prepared)
            match_threshold = st.slider(
                "Match threshold", 0.0, 100.0, DEFAULT_MATCH_THRESHOLD, disabled=prepared
            )
            verify_screen_presence = st.checkbox(
                "Verify on-screen presence (stage 6)", value=True, disabled=prepared
            )
            show_candidates = st.checkbox(
                "Extract other-candidate preview frames", value=True, disabled=prepared
            )
            keep_work_files = st.checkbox(
                "Keep downloaded video/audio after reset", value=False, disabled=prepared
            )
            save_session_log = st.checkbox(
                "Append every search to output/<video_id>/session.log", value=True, disabled=prepared
            )

        submitted = st.form_submit_button("Load video", disabled=prepared, width="stretch")

    if submitted and video_url.strip():
        config = PipelineConfig(
            video_url=video_url.strip(),
            target_text=None,
            language=language,
            engine=engine,
            whisper_model=whisper_model,
            device=device,
            match_threshold=match_threshold,
            verify_screen_presence=verify_screen_presence,
            extract_candidate_previews=show_candidates,
            save_session_log=save_session_log,
            keep_work_files=keep_work_files,
        )
        try:
            with st.spinner(
                "Downloading and transcribing -- this can take a few minutes on first run "
                "(model download, then ASR)..."
            ):
                pipeline = build_pipeline(config)
                session = pipeline.prepare()
        except PipelineError as exc:
            st.error(f"Could not prepare this video: {exc}")
        else:
            st.session_state["pipeline"] = pipeline
            st.session_state["session"] = session
            st.session_state["config"] = config
            st.session_state["history"] = []
            st.rerun()

    if prepared:
        session = st.session_state["session"]
        st.success(f'Loaded: "{session.video.title}"')
        st.caption(
            f"video_id={session.video.video_id} · "
            f"{session.video.duration_seconds:.0f}s · {session.video.fps:.1f}fps"
        )
        if st.button("🔄 New video / reset", width="stretch"):
            _reset_session()
            st.rerun()

if "session" not in st.session_state:
    st.info("⬅️ Enter a video URL in the sidebar and click **Load video** to get started.")
else:
    pipeline = st.session_state["pipeline"]
    session = st.session_state["session"]
    config = st.session_state["config"]

    st.header("2. Search for a line of dialogue")
    with st.form("search_form"):
        target_text = st.text_input("Dialogue line", placeholder="Eg. My mind rebels at stagnation")
        search = st.form_submit_button("Search", width="stretch")

    if search and target_text.strip():
        try:
            with st.spinner("Matching, locating frame, verifying on-screen presence..."):
                summary = pipeline.locate_dialogue(session, target_text.strip())
                log_path = append_session_log(config, session.video.video_id, target_text.strip(), summary)
        except PipelineError as exc:
            st.error(f"Search failed: {exc}")
        else:
            st.session_state["history"].insert(0, (target_text.strip(), summary, log_path))

    history = st.session_state.get("history", [])
    if history:
        st.subheader("Latest result")
        _render_result(*history[0])

        if len(history) > 1:
            st.subheader(f"Earlier searches this session ({len(history) - 1})")
            for text, summary, log_path in history[1:]:
                with st.expander(f'"{text}" -- frame {summary.frame_number} @ {summary.timestamp}'):
                    _render_result(text, summary, log_path)
