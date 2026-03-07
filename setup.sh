#!/bin/bash
# Why: One-command setup to create venv, install deps, and validate local services.
set -euo pipefail

echo "🎣 Setting up Viral-Flow..."
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Check FFmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "❌ FFmpeg not found. Install with: brew install ffmpeg (Mac) or apt install ffmpeg (Linux)"
    exit 1
fi

# Check Ollama
if ! curl -s http://localhost:11434/api/tags &> /dev/null; then
    echo "⚠️  Ollama not running. Start it with: ollama serve"
fi

# Check Piper binary
if [ ! -f "piper/piper" ]; then
    echo "⚠️  Piper binary not found in /piper folder."
    echo "    Download from: https://github.com/rhasspy/piper/releases"
fi

mkdir -p input/footage output temp models
echo "✅ Setup complete. Run: python src/ui_main.py"

