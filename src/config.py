from pathlib import Path

# Why: Centralize constants/paths so all modules share one source of truth.

BASE_DIR = Path(__file__).parent.parent

PIPER_BINARY = BASE_DIR / "piper" / "piper"
PIPER_MODEL = BASE_DIR / "models" / "ryan-high.onnx"
PIPER_CONFIG = BASE_DIR / "models" / "ryan-high.onnx.json"

INPUT_DIR = BASE_DIR / "input"
FOOTAGE_DIR = INPUT_DIR / "footage"
OUTPUT_DIR = BASE_DIR / "output"
TEMP_DIR = BASE_DIR / "temp"

OLLAMA_HOST = "http://localhost:11434"
MODEL_DIRECTOR = "deepseek-v3.1:671b-cloud"
MODEL_CREATIVE = "qwen3.5:397b-cloud"
MODEL_VISION = "minimax-m2:cloud"
OLLAMA_TIMEOUT = 300

OUTPUT_WIDTH = 1080
OUTPUT_HEIGHT = 1920
OUTPUT_FPS = 30
OUTPUT_FORMAT = "mp4"
OUTPUT_CRF = 23

SILENCE_THRESHOLD_DB = -40
MIN_SILENCE_DURATION = 0.3
KEYFRAME_EVERY_N_SECONDS = 3

BRAND_STYLE = "outdoorsy, clean, trustworthy, high-retention fishing content"
PLATFORM_TARGET = "tiktok_reels"

