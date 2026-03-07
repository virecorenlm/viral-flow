"""
Local smoke test for Viral-Flow.

Why: Provide a quick, production-friendly preflight to validate FFmpeg, Ollama,
and Piper are reachable before running long renders.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict

from src.config import FOOTAGE_DIR, PIPER_BINARY, TEMP_DIR


def run_smoke_test() -> Dict[str, Any]:
    """
    Run environment checks for Viral-Flow.

    Returns:
        Dict of check results.
    """
    results: Dict[str, Any] = {"ok": True, "checks": {}}

    def check(name: str, ok: bool, detail: str) -> None:
        results["checks"][name] = {"ok": ok, "detail": detail}
        if not ok:
            results["ok"] = False

    ffmpeg_ok = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
    check("ffmpeg", ffmpeg_ok, "ffmpeg/ffprobe found on PATH" if ffmpeg_ok else "Missing ffmpeg and/or ffprobe on PATH")

    piper_ok = Path(PIPER_BINARY).exists()
    check("piper", piper_ok, f"Found Piper at {PIPER_BINARY}" if piper_ok else f"Missing Piper at {PIPER_BINARY}")

    footage_ok = FOOTAGE_DIR.exists() and any(p.suffix.lower() == ".mp4" for p in FOOTAGE_DIR.glob("*.mp4"))
    check("footage", footage_ok, "Found .mp4 clips" if footage_ok else f"No .mp4 clips in {FOOTAGE_DIR}")

    # Ollama check is optional here because it requires the dependency and running server.
    try:
        from src.ollama_client import OllamaClient

        client = OllamaClient()
        tags = client.healthcheck()
        check("ollama", True, "Ollama reachable; tags listed")
        results["ollama_tags"] = tags
    except Exception as e:
        check("ollama", False, f"Ollama check failed: {e}")

    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    return results


if __name__ == "__main__":
    print(json.dumps(run_smoke_test(), indent=2))

