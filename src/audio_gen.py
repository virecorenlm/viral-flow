"""
Generate narration audio and timing signals for editing.

Responsibilities:
1. Call Piper TTS via subprocess to convert script.txt → narration.wav
2. Parse Piper's stdout for word-level timestamps when available
3. Run silence detection on the generated audio using pydub
4. Return audio file path + list of silence gaps + word timings

Why: The whole edit plan depends on reliable audio duration, pauses for natural cuts,
and timestamps for captions.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

from src.config import (
    MIN_SILENCE_DURATION,
    PIPER_BINARY,
    PIPER_CONFIG,
    PIPER_MODEL,
    SILENCE_THRESHOLD_DB,
)


log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class WordTiming:
    """A single word timing span in seconds."""

    word: str
    start: float
    end: float


def _ensure_executable(path: Path) -> None:
    """
    Validate that an executable exists at the given path.

    Parameters:
        path: Expected executable path

    Returns:
        None
    """
    # Why: Piper is user-provided; we want a clear error before rendering starts.
    if not path.exists():
        raise FileNotFoundError(f"Piper binary not found at: {path}")


def _parse_piper_word_timings(stdout_text: str) -> List[WordTiming]:
    """
    Attempt to parse word-level timings from Piper stdout.

    Piper timing output is not consistently documented across builds. This function:
    - Tries JSON-per-line parsing (common for JSON streaming tools)
    - Accepts several possible keys if present
    - Otherwise returns an empty list

    Parameters:
        stdout_text: Piper stdout text

    Returns:
        List of WordTiming (may be empty)
    """
    # Why: Keep parsing permissive so the pipeline still works if timings are missing.
    timings: List[WordTiming] = []
    for raw_line in stdout_text.splitlines():
        line = raw_line.strip()
        if not line or not (line.startswith("{") and line.endswith("}")):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        # Candidate locations for word timings (varies by Piper build/tooling wrappers).
        candidates = []
        if isinstance(obj, dict):
            for key in ("word_timings", "words", "timings", "alignment"):
                if key in obj and isinstance(obj[key], list):
                    candidates = obj[key]
                    break

        for item in candidates:
            if not isinstance(item, dict):
                continue
            word = str(item.get("word") or item.get("text") or "").strip()
            if not word:
                continue
            try:
                start = float(item.get("start") or item.get("start_time") or item.get("s"))
                end = float(item.get("end") or item.get("end_time") or item.get("e"))
            except (TypeError, ValueError):
                continue
            if end <= start:
                continue
            timings.append(WordTiming(word=word, start=start, end=end))

    # If multiple JSON lines contain partial data, ensure monotonic ordering.
    timings.sort(key=lambda w: (w.start, w.end))
    return timings


def _heuristic_word_timings(script_text: str, duration_seconds: float) -> List[WordTiming]:
    """
    Generate approximate word timings when the TTS engine does not provide alignment.

    Parameters:
        script_text: Full narration script
        duration_seconds: Total audio duration (seconds)

    Returns:
        List of WordTiming
    """
    # Why: Captions still need timings; this fallback keeps the editor functional offline.
    words = [w for w in script_text.replace("\n", " ").split(" ") if w.strip()]
    if not words or duration_seconds <= 0:
        return []

    # A simple proportional allocation based on word length.
    weights = [max(1, len(w)) for w in words]
    total_weight = float(sum(weights))
    t = 0.0
    out: List[WordTiming] = []
    for word, weight in zip(words, weights):
        span = (weight / total_weight) * duration_seconds
        start = t
        end = min(duration_seconds, t + span)
        out.append(WordTiming(word=word, start=start, end=end))
        t = end
    if out:
        out[-1] = WordTiming(word=out[-1].word, start=out[-1].start, end=duration_seconds)
    return out


def generate_narration_audio(script_text: str, output_wav_path: Path) -> Dict[str, Any]:
    """
    Run Piper TTS and analyze the resulting audio for silence and timing.

    Parameters:
        script_text: Narration script content (plain text)
        output_wav_path: Where to write the WAV file

    Returns:
        Dict with schema:
        {
          "audio_path": str,
          "duration_seconds": float,
          "silence_gaps": [{"start": float, "end": float}],
          "word_timings": [{"word": str, "start": float, "end": float}]
        }
    """
    _ensure_executable(PIPER_BINARY)
    output_wav_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(PIPER_BINARY),
        "--model",
        str(PIPER_MODEL),
        "--config",
        str(PIPER_CONFIG),
        "--output_file",
        str(output_wav_path),
        "--json-input",
    ]

    # Why: Use the exact subprocess pattern you specified for Piper invocation.
    process = subprocess.run(
        cmd,
        input=script_text,
        capture_output=True,
        text=True,
        timeout=120,
    )

    if process.returncode != 0:
        raise RuntimeError(
            "Piper failed.\n"
            f"STDERR:\n{process.stderr}\n"
            f"STDOUT:\n{process.stdout}\n"
        )

    # Why: pydub offers robust silence detection over decoded audio.
    from pydub import AudioSegment  # type: ignore
    from pydub.silence import detect_silence  # type: ignore

    audio = AudioSegment.from_file(output_wav_path)
    duration_seconds = len(audio) / 1000.0

    silence_ms = detect_silence(
        audio,
        min_silence_len=int(MIN_SILENCE_DURATION * 1000),
        silence_thresh=SILENCE_THRESHOLD_DB,
    )
    silence_gaps = [{"start": s / 1000.0, "end": e / 1000.0} for s, e in silence_ms]

    parsed = _parse_piper_word_timings(process.stdout)
    if not parsed:
        log.warning(
            "piper_word_timings_missing",
            note="Falling back to heuristic timings based on audio duration.",
        )
        parsed = _heuristic_word_timings(script_text, duration_seconds)

    return {
        "audio_path": str(output_wav_path),
        "duration_seconds": float(duration_seconds),
        "silence_gaps": silence_gaps,
        "word_timings": [
            {"word": w.word, "start": float(w.start), "end": float(w.end)} for w in parsed
        ],
    }


def generate_from_script_file(script_path: Path, temp_dir: Path) -> Dict[str, Any]:
    """
    Convenience helper: read a script file and generate narration to temp.

    Parameters:
        script_path: Path to script.txt
        temp_dir: Temp directory root

    Returns:
        Same schema as generate_narration_audio()
    """
    # Why: Keep UI code minimal; audio generation belongs in this module.
    script_text = script_path.read_text(encoding="utf-8").strip()
    if not script_text:
        raise ValueError("Script is empty.")
    return generate_narration_audio(script_text, temp_dir / "narration.wav")

