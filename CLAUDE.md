# CLAUDE.md — Viral-Flow

AI assistant guide for the Viral-Flow codebase. Read this before making changes.

---

## What This Project Is

Viral-Flow is a **fully local AI video editing suite** for Shorette's Bait and Tackle.
Drop raw fishing footage (and optionally a narration script) and it outputs a
**9:16 viral-ready vertical video** for TikTok / Instagram Reels.

Everything runs locally — the only network dependency is Ollama at `http://localhost:11434`.

---

## Pipeline Overview

There are two modes depending on whether the user provides a script:

### Mode A — User provides script
```
script.txt + footage/*.mp4
    → Piper TTS (narration.wav + word timings)
    → FFmpeg + Vision model (keyframe analysis)
    → Director LLM (edit plan JSON)
    → MoviePy + Pillow (render + captions)
    → output/viral_flow_YYYYMMDD_HHMMSS.mp4
```

### Mode B — Auto-generate script from footage (no script uploaded)
```
footage/*.mp4
    → FFmpeg + Vision model (keyframe analysis)  ← shared with step 3 below
    → Creative LLM (writes narration script)
    → Piper TTS (narration.wav + word timings)
    → Director LLM (edit plan JSON)             ← reuses clip_metadata from step 1
    → MoviePy + Pillow (render + captions)
    → output/viral_flow_YYYYMMDD_HHMMSS.mp4
```

In Mode B the vision model runs **once** — `generate_script_from_footage()` returns
`clip_metadata` which is passed directly to `render_from_plan()`, skipping the second
`analyze_footage()` call.

---

## Module Guide

| File | Role |
|---|---|
| `src/config.py` | All constants (paths, model names, resolution, thresholds). Change models here. |
| `src/ollama_client.py` | All Ollama calls. Enforces JSON contract with retry logic. |
| `src/audio_gen.py` | Piper TTS subprocess + silence detection + word timing extraction. |
| `src/scene_analyzer.py` | FFmpeg keyframe extraction + per-frame vision model analysis. |
| `src/script_writer.py` | Vision pre-watch → creative LLM writes Piper-ready narration. |
| `src/video_logic.py` | Director LLM edit plan + MoviePy render + Pillow caption overlay. |
| `src/ui_main.py` | Gradio UI, background job thread, pipeline orchestration. |
| `src/smoke_test.py` | Pre-flight checks for FFmpeg, Piper, Ollama. Run before debugging. |

---

## Ollama Models (`src/config.py`)

| Constant | Default | Role |
|---|---|---|
| `MODEL_DIRECTOR` | `deepseek-v3.1:671b-cloud` | Edit plan JSON (timeline, effects, transitions) |
| `MODEL_CREATIVE` | `qwen3.5:397b-cloud` | Caption plan JSON + auto-generated narration scripts |
| `MODEL_VISION` | `minimax-m2:cloud` | Per-frame scene analysis (requires vision capability) |

To swap a model, change the constant in `config.py` — nothing else needs updating.

---

## Key Conventions

### JSON enforcement
`OllamaClient.chat_json()` retries once on `JSONDecodeError` before raising.
Always use `chat_json()` for structured outputs and `chat()` directly for plain text
(like `write_script()` which returns prose, not JSON).

### Graceful degradation
Every LLM call has a fallback:
- Director fails → `_fallback_edit_plan()` in `video_logic.py` (energy-based pacing)
- Captions fail → `_fallback_caption_plan()` (5-word grouping)
- Piper timing missing → `_heuristic_word_timings()` in `audio_gen.py`

Never let an LLM failure crash the render. Wrap in try/except and use the fallback.

### Dependency injection
`OllamaClient` is instantiated once in `_pipeline_worker()` and passed down to
`analyze_footage()`, `generate_script_from_footage()`, and `render_from_plan()`.
Do not import or instantiate `OllamaClient` inside individual helper modules —
it makes unit testing impossible.

### No ImageMagick
Captions are rendered with Pillow (`_pillow_text_image()` in `video_logic.py`).
Do not introduce MoviePy's `TextClip` or any ImageMagick dependency.

### Subprocess for FFmpeg
Use `subprocess.run()` / `subprocess.check_output()` directly for FFmpeg/ffprobe.
Do not use MoviePy's FFmpeg wrappers for frame extraction or probing.

### clip_metadata reuse
When `generate_script_from_footage()` runs in auto-script mode, it returns
`(script_text, clip_metadata)`. The caller (`_pipeline_worker`) passes this
`clip_metadata` straight to `render_from_plan()`. Do not call `analyze_footage()`
again — the vision model is slow and expensive.

---

## Data Contracts

### `audio_data` dict (from `audio_gen.generate_narration_audio`)
```python
{
    "audio_path": str,
    "duration_seconds": float,
    "silence_gaps": [{"start": float, "end": float}],
    "word_timings": [{"word": str, "start": float, "end": float}],
}
```

### `clip_metadata` list item (from `scene_analyzer.analyze_footage`)
```python
{
    "clip_filename": str,
    "clip_path": str,
    "duration_seconds": float,
    "dominant_scene_type": str,   # casting|reeling|catch|lure_closeup|water|...
    "avg_visual_energy": float,   # 1–10
    "keep_ratio": float,          # fraction of frames marked keep=True
    "thumbnail_frame": str,       # path to best JPG frame
    "frames": [
        {
            "image_path": str,
            "scene_type": str,
            "subject": str,
            "visual_energy": int,
            "motion_level": str,  # static|slow|dynamic|fast
            "best_for": str,      # hook|action_moment|product_showcase|transition|background
            "keep": bool,
            "reason": str,
        }
    ],
}
```

### Director edit plan (JSON from `MODEL_DIRECTOR`)
```python
{
    "total_duration": float,
    "hook_clip": {"clip_filename": str, "start": float, "end": float, "reason": str},
    "timeline": [
        {
            "order": int,
            "clip_filename": str,
            "clip_start": float,
            "clip_end": float,
            "audio_start": float,
            "audio_end": float,
            "framing": "crop_center|blur_pad",
            "effect": "none|ken_burns_in|ken_burns_out|zoom_punch",
            "transition": "cut|fade|smash_cut",
            "transition_duration": float,
            "reason": str,
        }
    ],
    "caption_style": "bold_impact|clean_lower_third",
    "thumbnail_clip": str,
    "estimated_retention_score": int,  # 1–10
}
```

---

## Auto-Script: Vision-to-Piper Workflow

`src/script_writer.py` implements the vision-to-narration pipeline:

```python
script_text, clip_metadata = generate_script_from_footage(
    footage_dir=FOOTAGE_DIR,
    temp_dir=TEMP_DIR,
    ollama_client=client,
    style="Outdoorsy",        # Outdoorsy | Hype | Calm
    platform="TikTok",        # TikTok | Reels | Shorts
    max_length_seconds=60.0,
    save_to=INPUT_DIR / "script.txt",   # optional — persists for user review
)
```

Internally:
1. `analyze_footage()` runs `MODEL_VISION` on every keyframe (1 frame / 3s).
2. `ollama_client.write_script()` runs `MODEL_CREATIVE` with the structured
   frame descriptions and produces ~150 WPM narration prose.
3. The script is optionally saved to `input/script.txt` so the user can inspect
   or edit it before re-running.

The generated script is plain text — no JSON, no markdown, no stage directions.
Piper TTS ingests it directly via stdin.

---

## Adding a New Ollama Call

1. Add the method to `OllamaClient` in `src/ollama_client.py`.
2. Use `chat_json()` for structured JSON outputs, `chat()` for prose.
3. Accept `ollama_client` as a parameter in the calling function — never import
   and instantiate `OllamaClient` inside helpers.
4. Add a fallback that works without the model if it returns garbage.

---

## Video Effects Reference (`src/video_logic.py`)

| Effect | Behavior |
|---|---|
| `ken_burns_in` | Zoom 1.0 → 1.05 over clip duration |
| `ken_burns_out` | Zoom 1.05 → 1.0 over clip duration |
| `zoom_punch` | +8% scale at t=0, exponential decay back to 1.0 (~0.15s) |
| `none` | No transform |

| Framing | Behavior |
|---|---|
| `crop_center` | Scale to height, center-crop to 1080×1920 |
| `blur_pad` | Full-frame blur background + sharp foreground overlay |

| Transition | Behavior |
|---|---|
| `cut` | Instant cut |
| `fade` | Crossfade with `transition_duration` |
| `smash_cut` | Hard cut (optional zoom_punch on incoming clip) |

---

## Running Locally

```bash
# Setup (first time)
./setup.sh          # Mac/Linux
.\setup.ps1         # Windows

# Pre-flight check
python src/smoke_test.py

# Launch UI
source venv/bin/activate
python src/ui_main.py
# → http://localhost:7860
```

Required system binaries (must be on PATH): `ffmpeg`, `ffprobe`
Required local services: Ollama at `http://localhost:11434`
Required user-provided files: `piper/piper`, `models/ryan-high.onnx`, `models/ryan-high.onnx.json`

---

## Directory Layout

```
viral-flow/
├── src/
│   ├── config.py           ← change model names / paths / resolution here
│   ├── ollama_client.py    ← all LLM/VLM calls
│   ├── audio_gen.py        ← Piper TTS + silence + word timings
│   ├── scene_analyzer.py   ← FFmpeg keyframes + vision analysis
│   ├── script_writer.py    ← vision pre-watch → auto narration script
│   ├── video_logic.py      ← Director plan + MoviePy render + captions
│   ├── ui_main.py          ← Gradio UI + pipeline orchestration
│   └── smoke_test.py       ← environment preflight
├── input/
│   ├── script.txt          ← user script (or auto-generated here)
│   └── footage/            ← raw .mp4 clips (gitignored)
├── output/                 ← rendered MP4s (gitignored)
├── temp/                   ← intermediate files, auto-cleaned (gitignored)
├── piper/                  ← Piper TTS binary (gitignored, user provides)
├── models/                 ← ryan-high.onnx + .json (gitignored, user provides)
├── requirements.txt
├── setup.sh / setup.ps1
└── README.md
```

---

## What Is Gitignored

`output/`, `temp/`, `piper/`, `models/`, `input/footage/`, `input/script.txt`,
`.env*`, `__pycache__/`, venv directories, IDE files, log files.

Never commit binary model files, rendered videos, or secrets.
