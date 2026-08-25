cat > /home/claude/video-dialogue-finder/approach.md << 'DOCEOF'
# Approach: Locating a Dialogue's Frame in a Video (Audio-First Pipeline)

## 1. Problem statement (restated)

Given a video URL and a target dialogue string, find the exact video frame
where that dialogue is spoken, and return:
- the timestamp (HH:MM:SS.sss)
- the frame number
- the extracted/matched text
- the corresponding frame, saved as an image

**Key constraint driving the architecture:** the dialogue is not a
burned-in on-screen caption — it exists only in the audio track. This rules
out OCR as the core mechanism and makes **speech-to-text with word-level
timestamp alignment** the right tool: transcribe the audio, locate the
target phrase in the transcript, and map its timestamp to a video frame.

---

## 2. Problem statement → sprint breakdown

**Window:** Aug 24 (evening, post pre-placement talk) → Aug 26, 23:59:59 IST (~40 hrs)

### Sprint 1 — Ingestion & raw data flow (Aug 24 eve → Aug 25 AM)
| Task | Exit criterion |
|---|---|
| Download video via `yt-dlp` (ok.ru / YouTube / general) | Local video file + metadata (fps, duration, codec) confirmed against the real URL |
| Extract audio track via `ffmpeg` | Clean mono 16kHz WAV, correct duration |
| Wire up WhisperX, run once on real audio | Raw word-level transcript JSON produced; spot-checked by eye near where the dialogue is expected |

### Sprint 2 — Matching & frame extraction (Aug 25 daytime)
| Task | Exit criterion |
|---|---|
| Sliding-window fuzzy matcher over transcript words | Correctly locates the target phrase's start time on the real transcript |
| Timestamp → frame-number mapping | `frame_number = round(start_time * fps)`, verified against video metadata |
| Frame extraction via OpenCV seek | Correct frame image saved, visually confirmed to be the right moment |
| **End-to-end CLI run** | URL + target text → correct timestamp/frame/image on the real assignment video |

### Sprint 3 — Robustness & ambiguity handling (Aug 25 eve → Aug 26 early)
| Task | Exit criterion |
|---|---|
| No-match / low-confidence handling | Best candidate still returned, flagged `is_uncertain` with reason, never silent failure |
| Config knobs (match threshold, window size, `--language`) | Language param demonstrably swappable without touching pipeline code |
| Unit tests for matcher + timestamp mapping | `pytest` passes on core logic |

### Sprint 4 — Packaging & submission (Aug 26)
| Task | Exit criterion |
|---|---|
| Dockerfile (ffmpeg + WhisperX deps baked in) | `docker compose up --build` works from a clean clone |
| Full clean-machine test | No "works on my machine" surprises |
| `approach.md` / `prompt.txt` / `README.md` finalized | All three in repo root, cross-referenced |
| Push to GitHub | Well before 23:59:59 IST — not at the deadline |

---

## 3. Architecture

```
video URL
   │
   ▼
┌─────────────────────────────┐
│ [1] Ingestion                │  yt-dlp
│     VideoDownloader          │  → local video file + metadata
└──────────────┬────────────────┘
               ▼
┌─────────────────────────────┐
│ [2] Audio extraction          │  ffmpeg
│     AudioExtractor            │  → mono 16kHz WAV
└──────────────┬────────────────┘
               ▼
┌─────────────────────────────┐
│ [3] Speech-to-text + align    │  WhisperX
│     TranscriptionEngine       │  → [{word, start, end, conf}, ...]
└──────────────┬────────────────┘
               ▼
┌─────────────────────────────┐
│ [4] Fuzzy phrase matching     │  RapidFuzz, sliding window
│     PhraseMatcher              │  → best-matching span + start_time
└──────────────┬────────────────┘
               ▼
┌─────────────────────────────┐
│ [5] Timestamp → frame          │  frame_number = round(start_time * fps)
│     FrameLocator (OpenCV)      │  → exact frame image
└──────────────┬────────────────┘
               ▼
┌─────────────────────────────┐
│ [6] Reporting                  │  SQLite row + result.json + frame.png
│     ResultStore                │
└─────────────────────────────┘
```

Each bracketed stage is one module behind one interface — see §5.

---

## 4. Why this approach (and what it trades off against)

| Approach | Precision on "exact frame" | Speed | Multi-language | Verdict |
|---|---|---|---|---|
| OCR on sampled frames | N/A — no visual text exists in this video | Wasted work scanning frames for nothing | Needs per-language Tesseract data | **Rejected**: wrong tool for audio-only dialogue |
| Whisper (reference) or faster-whisper alone | Segment-level timestamps only; word timestamps are interpolated and drift up to ~1s | faster-whisper is fast | 99 languages via `language=` param | Drift of ~1s can point at the wrong frame on a fast cut — not precise enough |
| **WhisperX (faster-whisper + wav2vec2 forced alignment)** | **Sub-100ms word-level timestamps** — a second pass aligns each word to the actual waveform | Fast, batched, CPU-usable with smaller models | 99 languages for transcription; dedicated aligner models for major languages, graceful fallback otherwise | **Chosen.** This is the gap between "roughly the right second" and "the exact frame" |
| Vosk | Word-level but lower precision than forced alignment | Very fast, tiny footprint | Multiple languages, separate small models | Fallback option if WhisperX is too heavy for the dev machine |

**Robustness bonus of the audio-first approach:** the problem statement
notes the evaluators may substitute a different video/dialogue. Because the
full transcript is generated once, a new target phrase only requires
re-running the fuzzy match — not re-processing the video.

**Future scope (documented, not built for the deadline):** a hybrid pass
using scene-cut detection (e.g. PySceneDetect) within the matched second to
avoid landing on a motion-blurred transition frame.

---

## 5. Source structure (SOLID, one responsibility per module)

```
video-dialogue-finder/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── approach.md              # this file
├── prompt.txt               # AI-assistance prompt log
├── README.md
├── src/
│   ├── main.py               # CLI entrypoint / composition root — the
│   │                          # ONLY file that knows which concrete
│   │                          # implementation backs each interface
│   ├── pipeline.py            # orchestrates the 6 stages; depends only
│   │                          # on abstract interfaces (Dependency Inversion)
│   │
│   ├── ingestion/
│   │   ├── base.py            # VideoDownloader(ABC)
│   │   └── ytdlp_downloader.py
│   │
│   ├── audio/
│   │   ├── base.py            # AudioExtractor(ABC)
│   │   └── ffmpeg_extractor.py
│   │
│   ├── transcription/
│   │   ├── base.py            # TranscriptionEngine(ABC) — returns
│   │   │                       # word-level {word, start, end, conf}
│   │   ├── whisperx_engine.py  # default: forced-alignment precision
│   │   └── vosk_engine.py      # optional lightweight fallback
│   │
│   ├── matching/
│   │   ├── base.py            # PhraseMatcher(ABC)
│   │   └── fuzzy_matcher.py    # RapidFuzz sliding-window implementation
│   │
│   ├── frame_locator/
│   │   ├── base.py            # FrameLocator(ABC)
│   │   └── opencv_locator.py   # timestamp → frame_number → seek → image
│   │
│   └── output/
│       ├── base.py            # ResultStore(ABC)
│       └── sqlite_store.py     # result.json + matched_frame.png + SQLite row
│
├── tests/
│   ├── test_fuzzy_matcher.py
│   └── test_frame_locator.py
│
├── work/                       # scratch: downloaded video, extracted audio
└── output/                     # result.json, matched_frame.png
```

**SOLID mapping, explicitly:**
- **Single Responsibility** — each module does exactly one stage; `pipeline.py`
  only orchestrates, never implements a stage itself.
- **Open/Closed** — adding `WhisperXEngine` → `VoskEngine`, or a future
  `insanely-fast-whisper` engine, means adding a new file, not editing
  existing ones.
- **Liskov Substitution** — any `TranscriptionEngine` implementation is
  interchangeable behind `pipeline.py`'s calls; same for every other `base.py`.
- **Interface Segregation** — each ABC exposes only the one method the
  pipeline needs (`download()`, `extract_audio()`, `transcribe()`,
  `match()`, `locate_frame()`, `save()`) — no bloated multi-purpose interfaces.
- **Dependency Inversion** — `pipeline.py` and `DialoguePipeline.__init__`
  take interfaces as constructor arguments; concrete classes are wired only
  in `main.py` (the composition root).

Every module file is expected to carry a short docstring stating *why* that
implementation was chosen (mirrors the tradeoff table in §4) so the code is
self-documenting for the interview walkthrough.

---

## 6. Ambiguity & uncertainty handling

- **No transcript span clears the match threshold** → pipeline still
  returns its best-scoring candidate span, flagged `is_uncertain: true`,
  with a note (e.g. "check target phrase wording" or "lower match_threshold").
- **Match found but ASR confidence on those words is low** → returned as the
  answer but flagged uncertain, since a low-confidence transcription
  matching by luck is more likely a false positive than a wrong frame choice.
- **Multiple spans score similarly high** (dialogue repeated in the video) →
  report the *first* occurrence by default (matches "first appears" in the
  problem statement) but log all candidates above threshold for manual review.

---

## 7. Known limitations (for interview discussion)

- Assumes dialogue is spoken clearly enough for ASR — heavy background
  music/noise would degrade word-level alignment accuracy; not handled by
  a dedicated denoising stage in the current scope.
- WhisperX's forced-alignment model is per-language; languages without a
  dedicated wav2vec2 aligner fall back to Whisper's own (less precise)
  timestamps — documented as a graceful degradation, not a silent failure.
- No visual verification pass (e.g. confirming the extracted frame isn't
  mid-blink or mid-transition) — listed under future scope in §4 rather
  than built, to protect the deadline.
DOCEOF
