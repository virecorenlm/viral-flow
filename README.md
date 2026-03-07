# Viral-Flow — AI Video Editor (Local)

Viral-Flow is a **fully local** AI-powered video editing suite built for **Shorette's Bait and Tackle**.  
Drop a narration script and raw fishing footage, and it outputs a **viral-ready 9:16** vertical video for TikTok / Instagram Reels.

## Requirements

- **Python 3.11**
- **FFmpeg** installed as a **system binary** (`ffmpeg` and `ffprobe` on PATH)
- **Ollama** running locally at `http://localhost:11434`
- **Piper** TTS binary in `piper/` and the **Ryan-High** model files in `models/`

## Directory Layout

```
viral-flow/
├── piper/                        # Piper TTS binary lives here (user provides)
│   └── piper                     # executable
├── models/                       # Piper voice model files
│   ├── ryan-high.onnx
│   └── ryan-high.onnx.json
├── input/
│   ├── script.txt                # User's narration script
│   └── footage/                  # Raw .mp4 clips dropped here
├── output/                       # Final rendered videos saved here
├── temp/                         # Intermediate files, auto-cleaned after render
├── src/
│   ├── audio_gen.py
│   ├── video_logic.py
│   ├── scene_analyzer.py
│   ├── ollama_client.py
│   ├── ui_main.py
│   └── config.py
├── requirements.txt
├── setup.sh
└── README.md
```

## Ollama Models

Pull the models you requested:

```bash
ollama pull deepseek-v3.1:671b-cloud
ollama pull qwen3.5:397b-cloud
ollama pull minimax-m2:cloud
```

## Setup (Mac/Linux)

```bash
chmod +x setup.sh
./setup.sh
```

## Setup (Windows)

This repository includes `setup.sh` for Mac/Linux and `setup.ps1` for Windows.

Windows PowerShell:

```powershell
.\setup.ps1
```

If PowerShell blocks scripts, run PowerShell as your user and allow local scripts:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

FFmpeg must be available on PATH for both Windows and WSL.

## Run

```bash
source venv/bin/activate
python src/ui_main.py
```

Then open:

- `http://localhost:7860`

## How It Works (Pipeline)

- **TTS**: `src/audio_gen.py` runs Piper to generate `temp/narration.wav`, detects silence gaps, and produces word timings.
  - If Piper timing data is not present in stdout, Viral-Flow falls back to **heuristic word timings** so captions still work.
- **Scene analysis**: `src/scene_analyzer.py` uses FFmpeg to extract keyframes and asks the vision model for structured JSON.
- **Director plan**: `src/video_logic.py` asks the Director model for an edit timeline JSON.
  - If the Director model call fails, Viral-Flow falls back to an energy-based pacing plan.
- **Render**: MoviePy assembles the timeline into a vertical video and burns captions rendered via Pillow (no ImageMagick).

## Output

Final videos are written to:

- `output/viral_flow_YYYYMMDD_HHMMSS.mp4`

