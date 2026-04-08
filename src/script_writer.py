"""
Vision-to-script: Pre-watch footage with the Ollama vision model, then write a narration.

Why: Enables fully automated video creation — no manual script needed. The vision model
analyzes every keyframe of every clip, then the creative model writes a natural narration
that Piper TTS reads aloud and the Director cuts to.

Key design decision: analyze_footage() is called here and its clip_metadata is returned
alongside the script so the main pipeline can REUSE it for the Director edit plan without
re-running the vision model a second time. This halves Ollama calls in auto-script mode.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import structlog

from src.scene_analyzer import analyze_footage


log = structlog.get_logger(__name__)


def generate_script_from_footage(
    footage_dir: Path,
    temp_dir: Path,
    ollama_client: Any,
    style: str = "Outdoorsy",
    platform: str = "TikTok",
    max_length_seconds: float = 60.0,
    save_to: Optional[Path] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Pre-watch footage with the vision model and write a Piper-ready narration script.

    Pipeline inside this function:
    1. analyze_footage() — FFmpeg extracts keyframes; vision model describes each frame.
    2. ollama_client.write_script() — creative model writes narration from frame analysis.
    3. Optionally save the script to disk so the user can review/edit before re-running.

    Parameters:
        footage_dir: Directory containing .mp4 clips
        temp_dir: Temp directory root (keyframe JPGs go here)
        ollama_client: OllamaClient instance (needs .vision() and .write_script())
        style: Content style (Outdoorsy|Hype|Calm)
        platform: Target platform (TikTok|Reels|Shorts)
        max_length_seconds: Target video length — controls script word count
        save_to: If given, the generated script is written to this path so the user
                 can inspect or hand-edit it before the next run

    Returns:
        (script_text, clip_metadata)
        — script_text: plain narration ready for Piper TTS
        — clip_metadata: the analyze_footage() output; pass this directly to
          render_from_plan() so the vision model is NOT called a second time
    """
    log.info(
        "script_writer.start",
        footage_dir=str(footage_dir),
        style=style,
        platform=platform,
        max_length_seconds=max_length_seconds,
    )

    # Step 1: Vision model pre-watches all clips.
    clip_metadata = analyze_footage(
        footage_dir=footage_dir,
        temp_dir=temp_dir,
        ollama_client=ollama_client,
    )
    log.info("script_writer.footage_analyzed", clips=len(clip_metadata))

    # Step 2: Creative model writes the narration from the visual analysis.
    script = ollama_client.write_script(
        clip_metadata=clip_metadata,
        style=style,
        platform=platform,
        max_length_seconds=max_length_seconds,
    )
    if not script:
        raise RuntimeError(
            "Script generation returned empty text. "
            "Check that the creative Ollama model is running and responding."
        )
    log.info("script_writer.script_ready", words=len(script.split()))

    # Step 3: Persist for user inspection (optional).
    if save_to is not None:
        save_to.parent.mkdir(parents=True, exist_ok=True)
        save_to.write_text(script, encoding="utf-8")
        log.info("script_writer.saved", path=str(save_to))

    return script, clip_metadata
