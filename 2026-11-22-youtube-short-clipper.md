# YouTube Short Clipper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
> **MIT Reference Note:** Selected architectural ideas (per-object JSON rescue, optional direction sanitization, hook/claim verification, test-frame GPU encoder probe, and ffmpeg stall watchdog) are inspired by MIT-licensed `jipraks/yt-short-clipper`. Write original code; if any snippet is copied verbatim, include the MIT copyright notice.

**Goal:** Build the local "YouTube Short Clipper" web app that takes one YouTube URL (with optional user direction and output language) and produces up to 5 upload-ready packages (9:16 MP4 <=60s with burned-in ASS subtitles and optional 5s hook text in one encode, 3 verified title variants, 3 selectable 1080x1920 PNG thumbnails, and source-credit description).

**Architecture:** A FastAPI backend persists job/candidate/clip state in SQLite, executes analysis and clip-packaging pipelines in background threads, transcribes locally with `faster-whisper`, detects faces via bundled YuNet ONNX (Haar fallback), routes all `ffmpeg` invocations through one stderr-draining watchdog helper, and probes GPU encoders via a 1-frame test encode. A React + TypeScript + Tailwind + shadcn/ui frontend polls job and clip status across 4 Indonesian-language screens.

**Tech Stack:** Python 3.11+, FastAPI, Uvicorn, yt-dlp, ffmpeg, faster-whisper, OpenCV (`opencv-python-headless` + YuNet ONNX), Pillow, Pydantic v2, SQLite, Anthropic Python SDK, pytest; React, Vite, TypeScript, Tailwind CSS, shadcn/ui, Vitest.

**Spec:** `docs/superpowers/specs/2026-11-22-youtube-short-clipper-design.md`

## Global Constraints

- Python `>= 3.11`; Node `>= 20`; local execution only (no cloud server).
- LLM via official `anthropic` SDK; model from `ANTHROPIC_MODEL` (default `claude-sonnet-5-5`); API key from `.env` (`ANTHROPIC_API_KEY`); `.env.example` provided; `.env` and `data/jobs/` in `.gitignore`.
- **All LLM prompts written in English**; generated output text matches `output_language` (`"auto"` -> detected video language, or explicit code), while UI copy and scoring `reason` stay Indonesian.
- Temperature: `0.3` when user `direction` is present, SDK default when absent.
- Transcript always via `faster-whisper` (`WHISPER_MODEL` default `large-v3`, `WHISPER_DEVICE` default `auto`, `WHISPER_COMPUTE_TYPE` default `auto`: `float16` if CUDA available else `int8`; `word_timestamps=True`, `language=None`, `vad_filter=True`, `beam_size=5`). No YouTube captions.
- GPU encoder detection (`h264_nvenc`, `h264_amf`, `h264_qsv`, `h264_videotoolbox`) via 1-frame test encode, falling back to `libx264`. All `ffmpeg` calls use one helper with a stderr-draining thread and 300s stall watchdog.
- Face detection via shared `pipeline/faces.py` using bundled `face_detection_yunet_2023mar.onnx` (MIT license), falling back to Haar Cascade with warning log.
- Disk check: `min_free_disk_mb = 500` floor, scaled dynamically at `~2 GB per hour` (`max(500.0, (duration_s / 3600.0) * 2048.0)` MB).
- Candidate windows: `min_s=20`, `max_s=60`, sentence-boundary start/end; `< 20s` returns `[]` and marks job `failed` (`"Video terlalu pendek, minimal 20 detik"`). Explicit time range from `direction` is valid when `5s <= duration <= 60s` (exempt from `min_s=20`) and rejected (`422`) when `< 5s` or `> 60s`.
- Scoring: pre-select max `40` windows (`>= 15s` start gap), `10` per LLM call scored independently, rubric rewards conflict/confession/sharp claim/punchline/setup-buildup-payoff/standalone and penalizes filler/emotionless tech chatter; per-object `raw_decode` JSON rescue; request `target + 3` buffer when selecting highlights; top 5 discard `> 30%` overlap, with temporal spread enabled when `direction` is empty and disabled when `direction` is non-empty.
- Clip output: MP4 9:16 (`1080x1920`), max `60`s, single-pass encode with crop (`smart_crop` default, `center_crop`, `blur_fit`), subtitles (`"standard"` 2 lines x 32 chars or `"word_highlight"` 4 words/card UPPERCASE yellow active word), and optional `hook_text` (first 5s, <=15 words quote/sharp statement, no emoji, 3 words/line UPPERCASE in filled box).
- Claims in titles, hook text, and thumbnail text verified via LLM against transcript; flagged items regenerated once, clip job marked `failed` if still ungrounded.
- Anti-goals strictly forbidden: claims not in clip, copyright-evasion transforms (flip, pitch shift), secrets in repo.
- UI copy: 100% Indonesian, no fluff sentences, no em dash, no decorative emoji, responsive (`>= 44px` tap targets), keyboard accessible.

## Review Focus

- **Per-object `raw_decode` rescue & 10-item batch resilience:** A single corrupted or out-of-range JSON object among 10 scored windows must be discarded cleanly while preserving the 9 valid objects, and candidate requests ask for `target + 3` buffer before filtering. Pinned in Task 3 (`test_llm.py`) and Task 7 (`test_scoring.py`).
- **Direction sanitization, 1000-char limit & explicit time-range enforcement:** Template placeholders and injection block markers must be stripped, explicit ranges between `5s` and `60s` (e.g. `"2:00 - 2:10"`) must be accepted verbatim (exempt from `min_s=20`) and rejected with HTTP `422` when `< 5s` (e.g. `"2:00 - 2:03"`) or `> 60s`, direction must appear both before and after the transcript with `temperature=0.3`, and temporal spread must turn off when `direction` is non-empty. Pinned in Task 6 (`test_candidates.py`), Task 7 (`test_scoring.py`), and Task 12 (`test_api.py`).
- **Centralized `ffmpeg` stall watchdog (300s), GPU 1-frame probe & Windows `ass=...:fontsdir=...` escaping:** Hung `ffmpeg` processes must be killed by the watchdog after draining stderr on a separate thread; GPU encoder selection must rely on 1-frame encode exit status; Windows drive paths must be escaped for both ASS and `fontsdir`. Pinned in Task 1 (`test_ffmpeg_util.py`) and Task 9 (`test_render.py`).
- **Single-pass hook + `"Sorot Kata"` ASS + `smart_crop` 3-branch fallback:** `hook_text` (5s, 3 words/line, filled box) and `"word_highlight"` subtitles (4 words/card, yellow active word, `MarginV >= 300`) must render in one encode pass; `smart_crop` must fall back to `center_crop` (>50% faceless) or `blur_fit` (>35% x-spread). Pinned in Tasks 8 and 9.
- **Claim verification for titles, hook text & thumbnail text + async clip & HTTP Range 206:** Ungrounded claims after 1 regeneration must fail the background clip job (`status="failed"`); `GET /jobs/{id}/source` must return `206` with `Content-Range`. Pinned in Tasks 10, 11, and 12.

---

## File Structure

- `.gitignore`, `.env.example`, `README.md`
- `backend/requirements.txt`
- `backend/assets/fonts/Anton-Regular.ttf` & `OFL.txt`
- `backend/assets/models/face_detection_yunet_2023mar.onnx` & `LICENSE.txt`
- `backend/app/pipeline/ffmpeg_util.py` - `run_ffmpeg(cmd, stall_timeout_s=300.0)` with stderr thread & watchdog; `probe_video_encoders()`
- `backend/app/config.py` - env loading, `required_disk_mb()`, `is_cuda_available()`, `check_environment()`
- `backend/app/jobs.py` - SQLite repository (`jobs`, `candidates`, `clips`)
- `backend/app/llm.py` - Claude SDK client (`call_structured`, `call_structured_list` with `raw_decode` rescue, `temperature`)
- `backend/app/pipeline/models.py` - shared Pydantic domain models
- `backend/app/pipeline/download.py` - `download_source`
- `backend/app/pipeline/transcript.py` - `get_transcript` (`faster-whisper` only)
- `backend/app/pipeline/candidates.py` - `sanitize_direction`, `parse_explicit_time_range`, `build_windows`
- `backend/app/pipeline/scoring.py` - `preselect_windows`, `score_windows`
- `backend/app/pipeline/faces.py` - `detect_faces` (YuNet + Haar fallback)
- `backend/app/pipeline/render.py` - `render_clip` (`smart_crop`/`center_crop`/`blur_fit`, `"standard"`/`"word_highlight"`, single-pass `hook_text`)
- `backend/app/pipeline/titles.py` - `generate_titles`, `generate_hook_text`, claim verification
- `backend/app/pipeline/thumbnail.py` - `pick_thumbnail_frames`, `generate_thumbnail_text`, `render_thumbnail`
- `backend/app/main.py` - FastAPI app, startup check, background job & clip workers, REST & Range/asset endpoints
- `backend/tests/test_*.py` - unit & integration tests (one per module + `test_api.py`)
- `frontend/` - Vite + React + TypeScript + Tailwind + shadcn/ui with 4 pages and Vitest tests

---

### Task 1: Project Scaffolding, FFmpeg Runner, GPU Probe & Environment Checks

**Files:**
- Create: `.gitignore`, `.env.example`, `backend/requirements.txt`, `backend/app/__init__.py`, `backend/app/pipeline/__init__.py`, `backend/app/pipeline/models.py`, `backend/app/pipeline/ffmpeg_util.py`, `backend/app/config.py`
- Test: `backend/tests/test_config.py`, `backend/tests/test_ffmpeg_util.py`

**Interfaces:**
- Produces in `backend/app/pipeline/ffmpeg_util.py`:
  - `FFmpegError(RuntimeError)`, `FFmpegTimeoutError(FFmpegError)`
  - `run_ffmpeg(cmd: list[str], stall_timeout_s: float = 300.0) -> str`
  - `detect_video_encoder(ffmpeg_bin: str = "ffmpeg") -> tuple[str, list[str]]` (returns `(selected_encoder, working_hw_encoders)`)
- Produces in `backend/app/config.py`:
  - `Settings`: `anthropic_api_key`, `anthropic_model = "claude-sonnet-5-5"`, `whisper_model = "large-v3"`, `whisper_device = "auto"`, `whisper_compute_type = "auto"`, `data_dir`, `db_path`, `max_duration_s = 10800`, `min_free_disk_mb = 500`
  - `load_settings(env_file: Path | None = None) -> Settings`
  - `required_disk_mb(duration_s: float, min_free_disk_mb: int = 500) -> float`
  - `is_cuda_available() -> bool`
  - `EnvironmentStatus`: `ffmpeg_installed`, `ffmpeg_message`, `video_encoder`, `available_hw_encoders`, `api_key_configured`, `cuda_available`, `whisper_model`, `whisper_warning`, `disk_free_mb`, `disk_ok`, `ready`
  - `check_environment(settings: Settings) -> EnvironmentStatus`
- Produces in `backend/app/pipeline/models.py`:
  - `SourceInfo`, `WordTiming`, `Segment`, `TranscriptResult`, `Window`, `CriteriaScores`, `ScoredWindow`, `SubtitleLine`, `FaceBox`, `FrameCandidate`

- [ ] **Step 1: Initialize git repo, `.gitignore`, `.env.example`, and `backend/requirements.txt`**

Create `.gitignore`, `.env.example` (including `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL=claude-sonnet-5-5`, `WHISPER_MODEL=large-v3`, `WHISPER_DEVICE=auto`, `WHISPER_COMPUTE_TYPE=auto`), `backend/requirements.txt`, `git init`, and `backend/.venv`.

- [ ] **Step 2: Write failing tests in `backend/tests/test_ffmpeg_util.py` and `backend/tests/test_config.py`**

```python
# backend/tests/test_ffmpeg_util.py
import io
import time
import pytest
from app.pipeline import ffmpeg_util


def test_run_ffmpeg_watchdog_kills_stalled_process(monkeypatch):
    class HangingProc:
        def __init__(self):
            self.killed = False
            self.stdout = io.BytesIO(b"")
            self.stderr = self
            self.returncode = None

        def readline(self):
            while not self.killed:
                time.sleep(0.02)
            return b""

        def poll(self):
            return -9 if self.killed else None

        def kill(self):
            self.killed = True
            self.returncode = -9

        def wait(self, timeout=None):
            return self.returncode

    monkeypatch.setattr(ffmpeg_util.subprocess, "Popen", lambda *a, **kw: HangingProc())
    with pytest.raises(ffmpeg_util.FFmpegTimeoutError):
        ffmpeg_util.run_ffmpeg(["ffmpeg", "-i", "in.mp4", "out.mp4"], stall_timeout_s=0.15)


def test_detect_video_encoder_probes_with_test_frame_and_falls_back(monkeypatch):
    probed: list[str] = []

    def fake_run(cmd, stall_timeout_s=15.0):
        enc = cmd[cmd.index("-c:v") + 1]
        probed.append(enc)
        if enc == "h264_qsv":
            return ""
        raise ffmpeg_util.FFmpegError("not supported")

    monkeypatch.setattr(ffmpeg_util, "run_ffmpeg", fake_run)
    selected, hw = ffmpeg_util.detect_video_encoder("ffmpeg")
    assert probed == ["h264_nvenc", "h264_amf", "h264_qsv", "h264_videotoolbox"]
    assert selected == "h264_qsv"
    assert hw == ["h264_qsv"]
```

Also test in `backend/tests/test_config.py`: default settings, `required_disk_mb` (500 MB floor + 2048 MB/hour), missing `ffmpeg`, and CPU + `large-v3` `whisper_warning` + `video_encoder` field in `check_environment()`.

- [ ] **Step 3: Run tests to verify failure**

Run: `backend/.venv/Scripts/python -m pytest backend/tests/test_ffmpeg_util.py backend/tests/test_config.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 4: Implement `models.py`, `ffmpeg_util.py`, and `config.py`**

In `ffmpeg_util.py`, implement `run_ffmpeg` spawning `Popen`, draining `stderr` in a daemon `threading.Thread` updating `last_activity = time.monotonic()`, and polling until `proc.poll() is not None` or `(time.monotonic() - last_activity) > stall_timeout_s` (calling `proc.kill()` and raising `FFmpegTimeoutError`). Implement `detect_video_encoder` probing `("h264_nvenc", "h264_amf", "h264_qsv", "h264_videotoolbox")` with `-f lavfi -i color=c=black:s=256x256:d=0.1 -frames:v 1 -c:v <enc> -f null -` and falling back to `"libx264"`. Implement `config.py` and `models.py`.

- [ ] **Step 5: Run tests to verify pass**

Run: `backend/.venv/Scripts/python -m pytest backend/tests/test_ffmpeg_util.py backend/tests/test_config.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add .gitignore .env.example backend/requirements.txt backend/app/ backend/tests/test_config.py backend/tests/test_ffmpeg_util.py
git commit -m "chore: scaffold config, models, and ffmpeg watchdog/encoder probe"
```

---

### Task 2: SQLite Job Store (`backend/app/jobs.py`)

**Files:**
- Create: `backend/app/jobs.py`
- Test: `backend/tests/test_jobs.py`

**Interfaces:**
- Consumes: `Window`, `ScoredWindow`, `SourceInfo`, `Segment` from `app.pipeline.models`
- Produces: `DiskSpaceError`, `JobStore` (`create_job(url, direction=None, output_language="auto")`, `get_job`, `list_jobs`, `update_job`, `save_source_and_transcript`, `save_candidates`, `get_candidates`, `create_clip`, `update_clip`, `get_clip`, `list_clips_for_job`, `mark_interrupted_tasks_failed()`).

- [ ] **Step 1: Write failing tests in `backend/tests/test_jobs.py`**

Test job creation storing `direction` and `output_language`, stage progress updates, candidate persistence, async clip creation/updates (`queued` -> `titling` -> `rendering` -> `thumbnailing` -> `ready`/`failed` with `progress`, `error`, `subtitle_style`, `include_hook`, `hook_text`), `mark_interrupted_tasks_failed()` marking non-terminal jobs (`queued`, `downloading`, `transcribing`, `building_windows`, `scoring`) and non-terminal clips (`queued`, `titling`, `rendering`, `thumbnailing`) as `failed` with `"Terhenti karena aplikasi dimulai ulang. Ulangi proses."` while leaving `ready`/`failed` rows unchanged, and `DiskSpaceError` on SQLite disk-full operational error.

- [ ] **Step 2: Run test to verify failure**

Run: `backend/.venv/Scripts/python -m pytest backend/tests/test_jobs.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `backend/app/jobs.py`**

SQLite WAL schema with `jobs`, `candidates`, and `clips` tables.

- [ ] **Step 4: Run tests to verify pass**

Run: `backend/.venv/Scripts/python -m pytest backend/tests/test_jobs.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/jobs.py backend/tests/test_jobs.py
git commit -m "feat: add SQLite store for jobs, direction, candidates, and async clips"
```

---

### Task 3: Claude Structured LLM Client with Per-Object Rescue (`backend/app/llm.py`)

**Files:**
- Create: `backend/app/llm.py`
- Test: `backend/tests/test_llm.py`

**Interfaces:**
- Consumes: `Settings` from `app.config`
- Produces:
  - `MissingAPIKeyError(RuntimeError)`, `LLMValidationError(RuntimeError)`
  - `rescue_json_objects(raw_text: str, item_schema: type[T]) -> list[T]`
  - `LLMClient(api_key: str, model: str = "claude-sonnet-5-5", client: Any | None = None)`:
    - `call_structured(system_prompt: str, user_prompt: str, schema_model: type[T], max_retries: int = 2, temperature: float | None = None) -> T`
    - `call_structured_list(system_prompt: str, user_prompt: str, item_schema: type[T], max_retries: int = 2, temperature: float | None = None) -> list[T]`

- [ ] **Step 1: Write failing tests in `backend/tests/test_llm.py`**

```python
from types import SimpleNamespace
import pytest
from pydantic import BaseModel, Field
from app.llm import LLMClient, MissingAPIKeyError, LLMValidationError, rescue_json_objects


class DummyOut(BaseModel):
    score: float = Field(ge=0, le=10)


class BatchItem(BaseModel):
    window_id: str
    score: float = Field(ge=0, le=10)


def _fake_sdk(responses: list[str], calls: list[dict]):
    def create(**kwargs):
        calls.append(kwargs)
        text = responses[len(calls) - 1]
        return SimpleNamespace(content=[SimpleNamespace(text=text)])

    return SimpleNamespace(messages=SimpleNamespace(create=create))


def test_empty_api_key_raises():
    client = LLMClient(api_key="", model="claude-sonnet-5-5")
    with pytest.raises(MissingAPIKeyError):
        client.call_structured("sys", "usr", DummyOut)


def test_retries_twice_on_invalid_json_then_succeeds_and_passes_temperature():
    calls: list[dict] = []
    fake_sdk = _fake_sdk(["bukan json", '{"score": 15.0}', '```json\n{"score": 8.5}\n```'], calls)
    client = LLMClient(api_key="test-key", model="claude-sonnet-5-5", client=fake_sdk)
    result = client.call_structured("sys", "usr", DummyOut, max_retries=2, temperature=0.3)
    assert result.score == 8.5
    assert len(calls) == 3
    assert calls[0]["temperature"] == 0.3


def test_raises_validation_error_after_exhausting_retries():
    calls: list[dict] = []
    fake_sdk = _fake_sdk(["bad-1", "bad-2", "bad-3"], calls)
    client = LLMClient(api_key="test-key", model="claude-sonnet-5-5", client=fake_sdk)
    with pytest.raises(LLMValidationError):
        client.call_structured("sys", "usr", DummyOut, max_retries=2)
    assert len(calls) == 3


def test_rescue_json_objects_keeps_nine_valid_when_one_of_ten_is_broken():
    items = [f'{{"window_id": "w{i}", "score": {i}}}' for i in range(1, 10)]
    broken = '{"window_id": "w10", "score": "broken", oops}'
    raw = "[" + ", ".join(items[:4] + [broken] + items[4:]) + "]"
    rescued = rescue_json_objects(raw, BatchItem)
    assert len(rescued) == 9
    assert [r.window_id for r in rescued] == [f"w{i}" for i in range(1, 10)]
```

- [ ] **Step 2: Run test to verify failure**

Run: `backend/.venv/Scripts/python -m pytest backend/tests/test_llm.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `backend/app/llm.py`**

Implement `LLMClient.call_structured` (passing `temperature` only when not `None`) and `rescue_json_objects` + `call_structured_list`: first attempt full JSON array parse; if that fails or raises `ValidationError`, iterate through `{` positions in `raw_text` using `json.JSONDecoder().raw_decode`, validate each decoded dict with `item_schema.model_validate`, skip broken objects, and return all valid rescued objects (raising `LLMValidationError` only if 0 valid objects remain after retries).

- [ ] **Step 4: Run tests to verify pass**

Run: `backend/.venv/Scripts/python -m pytest backend/tests/test_llm.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/llm.py backend/tests/test_llm.py
git commit -m "feat: add Claude LLM client with temperature and per-object raw_decode rescue"
```

---

### Task 4: Source Video Downloader (`backend/app/pipeline/download.py`)

**Files:**
- Create: `backend/app/pipeline/download.py`
- Test: `backend/tests/test_download.py`

**Interfaces:**
- Consumes: `SourceInfo` from `app.pipeline.models`; `required_disk_mb` from `app.config`; `DiskSpaceError` from `app.jobs`
- Produces: `DownloadError(RuntimeError)`, `validate_youtube_url(url: str) -> str`, `download_source(url: str, work_dir: Path | None = None, max_duration_s: int = 10800, min_free_disk_mb: int = 500) -> SourceInfo`

- [ ] **Step 1: Write failing tests in `backend/tests/test_download.py`**

Test invalid URLs, playlist URLs, Shorts URLs, live videos, >3h duration, yt-dlp errors (private, age-restricted, 403 blocked), dynamic disk check (`duration_s=7200` rejected when free disk < 4096 MB), and `SourceInfo` returned without `captions_path` or caption download flags.

- [ ] **Step 2: Run test to verify failure**

Run: `backend/.venv/Scripts/python -m pytest backend/tests/test_download.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `backend/app/pipeline/download.py`**

Validate URL, `extract_info(..., download=False)`, verify `is_live` / `duration_s <= max_duration_s` / `required_disk_mb(duration_s, min_free_disk_mb)`, and download MP4 up to 1080p without subtitles.

- [ ] **Step 4: Run tests to verify pass**

Run: `backend/.venv/Scripts/python -m pytest backend/tests/test_download.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/pipeline/download.py backend/tests/test_download.py
git commit -m "feat: add YouTube URL validation, dynamic disk check, and yt-dlp downloader"
```

---

### Task 5: Faster-Whisper Transcript Extraction (`backend/app/pipeline/transcript.py`)

**Files:**
- Create: `backend/app/pipeline/transcript.py`
- Test: `backend/tests/test_transcript.py`

**Interfaces:**
- Consumes: `SourceInfo`, `Segment`, `WordTiming`, `TranscriptResult` from `app.pipeline.models`; `Settings`, `is_cuda_available` from `app.config`
- Produces: `resolve_whisper_runtime(settings: Settings) -> tuple[str, str]`, `get_transcript(source: SourceInfo, on_progress: Callable[[float, str], None] | None = None, settings: Settings | None = None) -> TranscriptResult`

- [ ] **Step 1: Write failing tests in `backend/tests/test_transcript.py`**

Test `resolve_whisper_runtime` (`float16` on CUDA vs `int8` on CPU), `WhisperModel` invocation (`word_timestamps=True, language=None, vad_filter=True, beam_size=5`), `TranscriptResult(segments, language="id")`, `Segment` having no `language` attribute, and incremental `on_progress` reporting.

- [ ] **Step 2: Run test to verify failure**

Run: `backend/.venv/Scripts/python -m pytest backend/tests/test_transcript.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `backend/app/pipeline/transcript.py`**

Always run `faster_whisper.WhisperModel` with configured device/compute_type, VAD filter, beam size 5, word timestamps, auto language detection, and progress updates.

- [ ] **Step 4: Run tests to verify pass**

Run: `backend/.venv/Scripts/python -m pytest backend/tests/test_transcript.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/pipeline/transcript.py backend/tests/test_transcript.py
git commit -m "feat: add faster-whisper transcript extractor with VAD and progress"
```

---

### Task 6: Window Builder, Direction Sanitizer & Time-Range Parser (`backend/app/pipeline/candidates.py`)

**Files:**
- Create: `backend/app/pipeline/candidates.py`
- Test: `backend/tests/test_candidates.py`

**Interfaces:**
- Consumes: `Segment`, `Window` from `app.pipeline.models`
- Produces:
  - `DirectionError(ValueError)`
  - `sanitize_direction(direction: str | None) -> str | None`
  - `parse_explicit_time_range(direction: str | None) -> tuple[float, float] | None`
  - `build_explicit_window(segments: list[Segment], start: float, end: float, language: str = "en") -> Window`
  - `build_windows(segments: list[Segment], min_s: float = 20.0, max_s: float = 60.0, language: str = "en") -> list[Window]`

- [ ] **Step 1: Write failing tests in `backend/tests/test_candidates.py`**

Test:
- Sentence-bounded `build_windows` (20-60s), punctuation-less >60s fallback split (`<= 60.0`s), and `< min_s` (e.g., 14s) returning `[]`.
- `sanitize_direction`: rejects/truncates `> 1000` chars (`DirectionError` if >1000 chars), strips template placeholders (`{{user_input}}`, `${var}`, `<% ... %>`) and instruction block markers (`<system>`, `</system>`, `[INST]`, `### SYSTEM`).
- `parse_explicit_time_range`: parses `"Fokus ke 2:00 - 2:50"` -> `(120.0, 170.0)`; parses `"2:00 - 2:10"` (10s, accepted and exempt from `min_s=20`) -> `(120.0, 130.0)`; parses `"01:05:10 - 01:05:55"` -> `(3910.0, 3955.0)`; raises `DirectionError` with clear Indonesian message when explicit range is below 5s (e.g. `"2:00 - 2:03"`), exceeds 60s (e.g. `"2:00 - 3:15"`), or is inverted (`"3:00 - 2:30"`).
- `build_explicit_window` slices transcript segments cleanly to `[start, end]` even when duration is between `5s` and `20s`.

- [ ] **Step 2: Run test to verify failure**

Run: `backend/.venv/Scripts/python -m pytest backend/tests/test_candidates.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `backend/app/pipeline/candidates.py`**

Implement `sanitize_direction`, `parse_explicit_time_range`, `build_explicit_window`, and sentence-bounded `build_windows`.

- [ ] **Step 4: Run tests to verify pass**

Run: `backend/.venv/Scripts/python -m pytest backend/tests/test_candidates.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/pipeline/candidates.py backend/tests/test_candidates.py
git commit -m "feat: add window builder, direction sanitizer, and explicit time-range parser"
```

---

### Task 7: Window Pre-Selection, English-Prompted Scoring & Temporal Spread (`backend/app/pipeline/scoring.py`)

**Files:**
- Create: `backend/app/pipeline/scoring.py`
- Test: `backend/tests/test_scoring.py`

**Interfaces:**
- Consumes: `Window`, `CriteriaScores`, `ScoredWindow` from `app.pipeline.models`; `LLMClient` from `app.llm`; `sanitize_direction` from `app.pipeline.candidates`
- Produces:
  - `preselect_windows(windows: list[Window], max_windows: int = 40, min_gap_s: float = 15.0) -> list[Window]`
  - `compute_weighted_total(scores: CriteriaScores) -> float`
  - `score_windows(windows: list[Window], llm_client: LLMClient | None = None, direction: str | None = None, target_count: int = 5, video_duration_s: float | None = None) -> list[ScoredWindow]`

- [ ] **Step 1: Write failing tests in `backend/tests/test_scoring.py`**

Test:
- Weight formula: `hook*0.30 + emotion*0.20 + surprise_or_value*0.20 + complete_idea*0.20 + standalone*0.10`.
- Synthetic 3-hour video (`300+` windows): `preselect_windows` returns at most `40` windows with `>= 15s` start gap; `score_windows` makes at most `4` LLM calls (`<= 10` windows/batch).
- Per-object JSON rescue in batch: when 1 of 10 scored objects in an LLM response is malformed JSON, `score_windows` keeps the 9 valid objects and still produces valid `ScoredWindow` results (using `target_count + 3` candidate buffer).
- Prompt inspection: system/user prompt is in **English**, includes the exact rubric cues (conflict, personal confession, sharp statement, punchline, setup/build-up/payoff, standalone; avoid filler & emotionless technical talk), instructs exclusion-rule adherence (`"lewati iklan"`), and when `direction` is provided: places sanitized direction in the prompt, **repeats it at the end after the transcript**, and passes `temperature=0.3` (vs `temperature=None` when `direction` is absent).
- Temporal spread & overlap:
  - **Without `direction` (`None` or empty):** given high-scoring windows clustered in minute 1 plus strong windows spread across early/mid/late thirds of a 30-minute video, final 5 picks discard `>30%` overlap and include selections distributed across the video duration.
  - **With non-empty `direction`:** temporal spread is disabled so final picks strictly follow highest-scoring windows aligned with the direction (even if all top 5 fall in the same part of the video), while still discarding `>30%` overlap.

- [ ] **Step 2: Run test to verify failure**

Run: `backend/.venv/Scripts/python -m pytest backend/tests/test_scoring.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `backend/app/pipeline/scoring.py`**

Implement `preselect_windows`, English rubric prompt builder (with repeated sanitized `direction` at end + `temperature=0.3`), `call_structured_list` with `target_count + 3` selection buffer and `raw_decode` per-object rescue, deterministic weighted total calculation, and `<= 30%` overlap selection (enabling temporal spread across video duration buckets only when `direction` is empty).

- [ ] **Step 4: Run tests to verify pass**

Run: `backend/.venv/Scripts/python -m pytest backend/tests/test_scoring.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/pipeline/scoring.py backend/tests/test_scoring.py
git commit -m "feat: add English-prompted scoring, direction handling, raw_decode rescue, and spread selection"
```

---

### Task 8: Shared Face Detection (`backend/app/pipeline/faces.py`)

**Files:**
- Create: `backend/assets/models/face_detection_yunet_2023mar.onnx`, `backend/assets/models/LICENSE.txt`, `backend/app/pipeline/faces.py`
- Test: `backend/tests/test_faces.py`

**Interfaces:**
- Consumes: `FaceBox` from `app.pipeline.models`
- Produces: `detect_faces(frame: np.ndarray) -> list[FaceBox]`

- [ ] **Step 1: Bundle YuNet ONNX model & MIT license in `backend/assets/models/`**

Download `face_detection_yunet_2023mar.onnx` and `LICENSE` (as `LICENSE.txt`) from `github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet`.

- [ ] **Step 2: Write failing tests in `backend/tests/test_faces.py`**

Test YuNet primary path returning `list[FaceBox]` and Haar Cascade fallback + `WARNING` log when ONNX loading fails.

- [ ] **Step 3: Run test to verify failure**

Run: `backend/.venv/Scripts/python -m pytest backend/tests/test_faces.py -v`
Expected: FAIL

- [ ] **Step 4: Implement `backend/app/pipeline/faces.py`**

Lazy-cached `cv2.FaceDetectorYN.create` with Haar fallback + `logging.warning`.

- [ ] **Step 5: Run tests to verify pass**

Run: `backend/.venv/Scripts/python -m pytest backend/tests/test_faces.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add backend/assets/models/ backend/app/pipeline/faces.py backend/tests/test_faces.py
git commit -m "feat: add shared YuNet face detector with Haar Cascade fallback"
```

---

### Task 9: 9:16 Video, Hook Box & Dual-Style ASS Renderer (`backend/app/pipeline/render.py`)

**Files:**
- Create: `backend/app/pipeline/render.py`
- Test: `backend/tests/test_render.py`

**Interfaces:**
- Consumes: `SourceInfo`, `Window`, `SubtitleLine`, `FaceBox` from `app.pipeline.models`; `detect_faces` from `app.pipeline.faces`; `run_ffmpeg`, `FFmpegError` from `app.pipeline.ffmpeg_util`
- Produces:
  - `RenderError(RuntimeError)`
  - `wrap_subtitle_events(subtitles: list[SubtitleLine], max_chars_per_line: int = 32, max_lines: int = 2) -> list[SubtitleLine]`
  - `build_word_highlight_events(segments: list[Segment], window_start: float, window_end: float, words_per_card: int = 4) -> list[tuple[float, float, str]]`
  - `build_hook_ass_events(hook_text: str, duration_s: float = 5.0, words_per_line: int = 3) -> list[str]`
  - `build_ass_content(...) -> str`
  - `escape_ffmpeg_filter_path(path: Path) -> str`
  - `resolve_smart_crop(video_path: Path, start: float, end: float, detector: Callable | None = None) -> tuple[Literal["smart_crop", "center_crop", "blur_fit"], int | None]`
  - `render_clip(source, window, subtitles, out_path, framing="smart_crop", subtitle_style="standard", hook_text=None, video_encoder="libx264") -> Path`

- [ ] **Step 1: Write failing tests in `backend/tests/test_render.py`**

Test:
- `"standard"` style: `wrap_subtitle_events` splits 110-char line into sequential cards <= 2 lines x 32 chars; `build_ass_content` uses `Fontname: Anton` and `MarginV >= 300` (outside bottom 15% safe zone).
- `"word_highlight"` style: groups Whisper `WordTiming` entries into **4 words per card**, **UPPERCASE**, highlighting only the active word in ASS yellow (`{\c&H00FFFF&}`) per word duration, keeping `MarginV >= 300`.
- `hook_text`: formats UPPERCASE, 3 words per line, inside filled background box (`BorderStyle=3`), active only in `0..5s`, embedded into the **same single-pass ffmpeg ASS/filtergraph** alongside crop and subtitles.
- `escape_ffmpeg_filter_path(Path("C:\\Users\\tester\\data\\subs.ass"))` produces `C\:/Users/tester/data/subs.ass`, and command uses `ass=<escaped_ass>:fontsdir=<escaped_fontsdir>` plus the passed `video_encoder`.
- `resolve_smart_crop` 3-branch tests (`smart_crop` median face x-center clamped, >50% faceless -> `center_crop`, >35% width spread -> `blur_fit`).
- Single-pass `ffmpeg` invocation routed through `ffmpeg_util.run_ffmpeg` (no `hflip`/`vflip`/`asetrate`/`rubberband` filters).
- Real `ffmpeg` integration test (NO `skipif`: fails if `ffmpeg` is missing) encoding a 2-second synthetic `testsrc` MP4 with hook + subtitles and verifying `1080x1920` via `ffprobe`.

- [ ] **Step 2: Run test to verify failure**

Run: `backend/.venv/Scripts/python -m pytest backend/tests/test_render.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `backend/app/pipeline/render.py`**

Implement standard & word-highlight ASS generators, 5s hook box ASS style (`HookBox` with `BorderStyle=3` + 3 words/line UPPERCASE), Windows filtergraph escaping (`ass=...:fontsdir=...`), `resolve_smart_crop`, and single-pass `run_ffmpeg` invocation with `video_encoder`.

- [ ] **Step 4: Run tests to verify pass**

Run: `backend/.venv/Scripts/python -m pytest backend/tests/test_render.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/pipeline/render.py backend/tests/test_render.py
git commit -m "feat: add single-pass 9:16 renderer with smart_crop, Sorot Kata ASS, and 5s hook box"
```

---

### Task 10: Grounded Titles & Hook Text Generator (`backend/app/pipeline/titles.py`)

**Files:**
- Create: `backend/app/pipeline/titles.py`
- Test: `backend/tests/test_titles.py`

**Interfaces:**
- Consumes: `Window` from `app.pipeline.models`; `LLMClient` from `app.llm`
- Produces:
  - `GroundedClaimError(RuntimeError)`
  - `generate_titles(window: Window, transcript_text: str, output_language: str | None = None, llm_client: LLMClient | None = None) -> list[str]`
  - `generate_hook_text(transcript_text: str, output_language: str, llm_client: LLMClient | None = None) -> str`

- [ ] **Step 1: Write failing tests in `backend/tests/test_titles.py`**

Test:
- English prompts instruct target output language (`output_language` or `window.language`), returning 3 titles `<= 70` chars.
- `generate_hook_text` enforces `<= 15` words, no emoji, direct quote or sharp statement from transcript (not summary), matching `output_language`.
- Claim verification pass for both titles and `hook_text`: flagged claim triggers 1 regeneration + re-verification; persistent ungrounded claim raises `GroundedClaimError`.

- [ ] **Step 2: Run test to verify failure**

Run: `backend/.venv/Scripts/python -m pytest backend/tests/test_titles.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `backend/app/pipeline/titles.py`**

Use English system/user prompts, Pydantic schemas (`TitlesResponse`, `HookTextResponse`, `ClaimCheckResponse`), 1-retry claim verification, and emoji stripping/rejection on `hook_text`.

- [ ] **Step 4: Run tests to verify pass**

Run: `backend/.venv/Scripts/python -m pytest backend/tests/test_titles.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/pipeline/titles.py backend/tests/test_titles.py
git commit -m "feat: add English-prompted grounded titles and 15-word hook generator with verification"
```

---

### Task 11: Thumbnail Picker & 1080x1920 PNG Renderer (`backend/app/pipeline/thumbnail.py`)

**Files:**
- Create: `backend/assets/fonts/Anton-Regular.ttf`, `backend/assets/fonts/OFL.txt`, `backend/app/pipeline/thumbnail.py`
- Test: `backend/tests/test_thumbnail.py`

**Interfaces:**
- Consumes: `FaceBox`, `FrameCandidate` from `app.pipeline.models`; `detect_faces` from `app.pipeline.faces`; `LLMClient` from `app.llm`; `GroundedClaimError` from `app.pipeline.titles`
- Produces:
  - `pick_thumbnail_frames(clip_path: Path, n: int = 3, out_dir: Path | None = None) -> list[FrameCandidate]`
  - `generate_thumbnail_text(transcript_text: str, output_language: str, llm_client: LLMClient | None = None) -> str`
  - `render_thumbnail(frame: Path, text: str, out_path: Path, focus_box: FaceBox | None = None, accent_shape: Literal["arrow", "circle"] = "arrow") -> Path`

- [ ] **Step 1: Bundle `Anton-Regular.ttf` and `OFL.txt` in `backend/assets/fonts/`**

Download from `github.com/google/fonts/tree/main/ofl/anton` and verify Pillow loads it cleanly.

- [ ] **Step 2: Write failing tests in `backend/tests/test_thumbnail.py`**

Test `pick_thumbnail_frames` returning `list[FrameCandidate]` (`faces.detect_faces` + Laplacian variance), `generate_thumbnail_text` (English prompt, `output_language`, 3-5 words, claim verification + 1x regen -> `GroundedClaimError`), and `render_thumbnail` (`1080x1920` PNG, cover-crop centered on `focus_box`, arrow/circle pointing to `focus_box` or upper-center when `None`).

- [ ] **Step 3: Run test to verify failure**

Run: `backend/.venv/Scripts/python -m pytest backend/tests/test_thumbnail.py -v`
Expected: FAIL

- [ ] **Step 4: Implement `backend/app/pipeline/thumbnail.py`**

Implement frame candidate selection, claim-verified thumbnail text generation, and Pillow `1080x1920` thumbnail rendering.

- [ ] **Step 5: Run tests to verify pass**

Run: `backend/.venv/Scripts/python -m pytest backend/tests/test_thumbnail.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add backend/assets/fonts/ backend/app/pipeline/thumbnail.py backend/tests/test_thumbnail.py
git commit -m "feat: add YuNet-aware frame picker, verified thumb text, and 1080x1920 renderer"
```

---

### Task 12: FastAPI Endpoints & Async Workers (`backend/app/main.py`)

**Files:**
- Create: `backend/app/main.py`
- Test: `backend/tests/test_api.py`

**Interfaces:**
- Consumes: `config.py`, `jobs.py`, `llm.py`, and all `pipeline/*.py` modules
- Produces HTTP endpoints:
  - `GET /health`
  - `POST /jobs` `{url, direction, output_language}`
  - `GET /jobs/{id}`, `GET /jobs/{id}/candidates`, `GET /jobs/{id}/source` (Range 206)
  - `POST /jobs/{id}/clips` `{candidate_id, start_offset, end_offset, subtitle_edits, framing, subtitle_style, include_hook, hook_text}` -> `{clip_id, status: "queued"}`
  - `GET /clips/{id}` (`status`, `progress`, `error`, plus assets when `ready`)
  - `GET /clips/{id}/video`, `GET /clips/{id}/thumbnails/{index}`, `GET /clips/{id}/description`

- [ ] **Step 1: Write failing tests in `backend/tests/test_api.py`**

Test:
- `GET /health` exposes `video_encoder`, `available_hw_encoders`, `cuda_available`, `whisper_model`, `whisper_warning`, `ffmpeg_installed`, `api_key_configured`, `disk_ok`.
- Startup recovery test: pre-populate SQLite DB with stuck non-terminal job (`transcribing`) and clip (`rendering`) plus terminal `ready` rows; creating/starting FastAPI app (`TestClient` lifespan) marks only the stuck non-terminal job and clip as `failed` with exact error `"Terhenti karena aplikasi dimulai ulang. Ulangi proses."`.
- `POST /jobs` rejects invalid URL/playlist/Shorts (`400`), `direction > 1000` chars (`422`), explicit time range `< 5s` e.g. `"2:00 - 2:03"` (`422`), or explicit time range `> 60s` (`422`).
- `POST /jobs` with explicit range `"2:00 - 2:10"` (10s, exempt from `min_s=20`) and `"2:00 - 2:45"` succeeds (`200`/`202`) and produces candidate at the exact timestamps.
- Video `< 20s` (without valid explicit range) marks job `failed` with `"Video terlalu pendek, minimal 20 detik"`.
- `GET /jobs/{id}/source` with `Range: bytes=0-1023` returns `206` with `Content-Range: bytes 0-1023/<total>`.
- `POST /jobs/{id}/clips` rejects invalid/over-60s offsets (`422`), returns `{clip_id, status: "queued"}` immediately on valid request, runs `queued -> titling -> rendering -> thumbnailing -> ready` (or `failed` on `GroundedClaimError`), and `GET /clips/{id}` returns full package with `hook_text` and source credit description.

- [ ] **Step 2: Run test to verify failure**

Run: `backend/.venv/Scripts/python -m pytest backend/tests/test_api.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `backend/app/main.py`**

Wire FastAPI app with lifespan startup calling `check_environment()` and `job_store.mark_interrupted_tasks_failed()`, CORS for `http://localhost:5173`, `ThreadPoolExecutor` workers for job analysis and clip generation (`titling` before `rendering` so verified `hook_text` burns in single pass), HTTP 422 mapping for `DirectionError`, Range `206` streaming, and asset endpoints.

- [ ] **Step 4: Run tests to verify pass**

Run: `backend/.venv/Scripts/python -m pytest backend/tests/test_api.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/main.py backend/tests/test_api.py
git commit -m "feat: add FastAPI endpoints, direction/language support, async clip worker, and Range 206"
```

---

### Task 13: Frontend Setup (Vite + React + TypeScript + Tailwind + shadcn/ui)

**Files:**
- Create: `frontend/package.json`, `frontend/index.html`, `frontend/tsconfig.json`, `frontend/vite.config.ts`, `frontend/components.json`, `frontend/src/index.css`, `frontend/src/types/api.ts`, `frontend/src/lib/api.ts`, `frontend/src/components/StudioShell.tsx`, `frontend/src/components/ui/*`

- [ ] **Step 1: Scaffold Vite React TS + Tailwind + shadcn/ui + Vitest**
- [ ] **Step 2: Configure typography, design tokens, and `StudioShell.tsx`** (`<html lang="id">`, `#121316` / `#1B1D22` / `#E11D48`, `Plus Jakarta Sans` + `JetBrains Mono`)
- [ ] **Step 3: Implement `frontend/src/types/api.ts` and `frontend/src/lib/api.ts`** matching `JobCreateRequest { url, direction?, output_language }`, `HealthStatus` (`video_encoder`, `cuda_available`, `whisper_warning`), `ClipCreateRequest` (`framing`, `subtitle_style`, `include_hook`, `hook_text`), and `ClipDetail` (`status`, `progress`, `error`).
- [ ] **Step 4: Run `cd frontend && npm run build`** (Expected: PASS)
- [ ] **Step 5: Commit**

```bash
git add frontend/
git commit -m "feat: scaffold React, Tailwind, shadcn/ui studio shell and API client"
```

---

### Task 14: Frontend 4 Screens & Tests

**Files:**
- Create: `frontend/src/pages/HomePage.tsx`, `frontend/src/pages/CandidatesPage.tsx`, `frontend/src/pages/EditorPage.tsx`, `frontend/src/pages/ResultPage.tsx`, `frontend/src/__tests__/app.test.tsx`
- Modify: `frontend/src/App.tsx`

- [ ] **Step 1: Write failing UI tests in `frontend/src/__tests__/app.test.tsx`**

Test:
1. `HomePage`: shows warnings (`ffmpeg_installed === false` and `!cuda_available && whisper_model === "large-v3"`), renders optional `"Arahan"` textarea (1000-char counter) and `"Bahasa Keluaran"` selector (`Otomatis`, `Indonesia (id)`, `Inggris (en)`, etc.), and submits `{ url, direction, output_language }`.
2. `CandidatesPage`: renders 5 ranked candidates, 5 criteria scores, 1-sentence reason, range preview, and empty/error states.
3. `EditorPage`: renders 3 framing options (`"Ikuti Wajah"` default, `"Potong Tengah"`, `"Utuh + Latar Blur"`), subtitle style selector (`"Standar"` vs `"Sorot Kata"`), toggle `"Tampilkan Teks Hook (5 Detik Pertama)"`, enforces `<= 60s`, polls `GET /clips/{id}` after clicking `"Render Paket Short"`, navigates to `/clips/:clipId` on `ready`, and shows error alert on `failed`.
4. `ResultPage`: displays MP4 download, 3 selectable titles, 3 selectable 1080x1920 thumbnails, and copyable source-credit description.

- [ ] **Step 2: Run `cd frontend && npm test -- --run`** (Expected: FAIL)
- [ ] **Step 3: Implement all 4 pages and wire `frontend/src/App.tsx`** (100% Indonesian, no em dash, no decorative emoji, `>= 44px` tap targets, empty/loading/error states)
- [ ] **Step 4: Run `cd frontend && npm test -- --run && npm run build`** (Expected: PASS)
- [ ] **Step 5: Commit**

```bash
git add frontend/src/
git commit -m "feat: implement 4-screen frontend workflow with direction, subtitle styles, and hook toggle"
```

---

### Task 15: README, Full Test Suite (0 Skipped) & Real YouTube Verification

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write `README.md`**

Include Indonesian setup instructions: Python 3.11+, `ffmpeg` (and hardware encoder notes), Node.js, `.env.example` -> `.env`, **GPU/CUDA & `WHISPER_MODEL` guide** (`large-v3` default untuk GPU NVIDIA/CUDA; `medium` atau `small` disarankan untuk CPU), running backend (`uvicorn app.main:app` untuk pemakaian normal, serta **peringatan bahwa `--reload` akan me-restart proses saat file berubah dan menghentikan job/clip yang sedang berjalan**), running frontend, and running tests.

- [ ] **Step 2: Run full automated test suites (0 skipped)**

Run:
- `backend/.venv/Scripts/python -m pytest backend/tests -v`
- `cd frontend && npm test -- --run && npm run build`
Confirm every test passes with **0 skipped** across the entire suite (including real `ffmpeg` integration test).

- [ ] **Step 3: Real YouTube link end-to-end verification**

Verify `ffmpeg` and `ANTHROPIC_API_KEY` availability locally, run the pipeline end-to-end against one real public YouTube link, generate a complete package (`clip.mp4`, 3 verified titles, optional verified hook, 3 PNG `1080x1920` thumbnails, `description.txt`), inspect outputs via `ffprobe` & Pillow, and report exact observed results.

- [ ] **Step 4: Final commit**

```bash
git add README.md
git commit -m "docs: add local setup, CUDA/WHISPER_MODEL guide, and run instructions in README"
```

---

### Task 16 (Opsional / Tahap Lanjut): Dynamic Face Tracking (`"dynamic_face"`)

> **STOP / GUARDRAIL:** Tugas ini adalah tahap lanjut opsional. **JANGAN dikerjakan sebelum seluruh 15 tugas utama di atas (Task 1 s.d. Task 15) lulus verifikasi penuh.**

**Files:**
- Modify: `backend/app/pipeline/models.py`, `backend/app/pipeline/render.py`, `backend/app/main.py`, `frontend/src/types/api.ts`, `frontend/src/pages/EditorPage.tsx`
- Test: `backend/tests/test_render.py`

**Interfaces:**
- Extends `framing` literal to `Literal["smart_crop", "center_crop", "blur_fit", "dynamic_face"]`.
- Produces in `backend/app/pipeline/render.py`:
  - `compute_dynamic_face_centers(centers: list[float | None], crop_w: int, frame_w: int, ema_alpha: float = 0.15, dead_zone_ratio: float = 0.02, snap_ratio: float = 0.35) -> list[int]`

- [ ] **Step 1: Write failing unit tests for `compute_dynamic_face_centers` in `backend/tests/test_render.py`**

Test:
1. Micro-movement within `<= 2%` of `crop_w` (dead zone) keeps crop center unchanged (no jitter).
2. Smooth shift between `2%` and `35%` of `crop_w` applies Exponential Moving Average (`alpha = 0.15`).
3. Large shift `> 35%` of `crop_w` snaps immediately to the new target center, clamped to `[crop_w // 2, frame_w - crop_w // 2]`.

- [ ] **Step 2: Run test to verify failure**
- [ ] **Step 3: Implement `compute_dynamic_face_centers`, ffmpeg piecewise/sendcmd crop filter integration, and 4th option `"Lacak Wajah Dinamis"` in `EditorPage.tsx`**
- [ ] **Step 4: Run full `pytest` + `npm test` + `npm run build` to verify pass**
- [ ] **Step 5: Commit**

```bash
git add backend/ frontend/
git commit -m "feat(optional): add dynamic_face framing mode with EMA smoothing and dead zone"
```
