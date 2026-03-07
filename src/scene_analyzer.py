"""
Footage scene analysis using FFmpeg keyframes + Ollama vision.

Responsibilities:
1. For each .mp4 in /input/footage/, extract 1 keyframe every N seconds using ffmpeg
2. Send each frame to minimax-m2:cloud via Ollama vision API
3. Return structured scene metadata for each clip

Why: The AI Director needs clip-level semantics (casting/catch/lure closeup, etc.)
to choose a strong hook and keep retention high.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import structlog

from src.config import KEYFRAME_EVERY_N_SECONDS


log = structlog.get_logger(__name__)


VISION_PROMPT = """You are analyzing fishing video footage. Describe this frame in detail.
Return ONLY valid JSON:
{
  "scene_type": "casting|reeling|catch|lure_closeup|water|boat|person_talking|b_roll|other",
  "subject": "what is the main subject",
  "visual_energy": 1-10,
  "motion_level": "static|slow|dynamic|fast",
  "best_for": "hook|action_moment|product_showcase|transition|background",
  "keep": true/false,
  "reason": "one sentence"
}
"""


@dataclass(frozen=True)
class FrameAnalysis:
    """Analysis for a single extracted frame."""

    image_path: str
    scene_type: str
    subject: str
    visual_energy: int
    motion_level: str
    best_for: str
    keep: bool
    reason: str


@dataclass(frozen=True)
class ClipMetadata:
    """Aggregated analysis for a single footage clip."""

    clip_filename: str
    clip_path: str
    duration_seconds: float
    dominant_scene_type: str
    avg_visual_energy: float
    keep_ratio: float
    thumbnail_frame: Optional[str]
    frames: List[Dict[str, Any]]


def _run_ffmpeg_extract_frames(clip_path: Path, out_pattern: Path) -> None:
    """
    Extract JPG frames from a clip at a fixed interval.

    Parameters:
        clip_path: Path to input .mp4
        out_pattern: Output filename pattern ending in *_frame_%04d.jpg

    Returns:
        None
    """
    # Why: Requirement explicitly says use ffmpeg subprocess (not MoviePy) for keyframes.
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(clip_path),
        "-vf",
        f"fps=1/{KEYFRAME_EVERY_N_SECONDS}",
        "-q:v",
        "2",
        str(out_pattern),
    ]
    subprocess.run(cmd, check=True)


def _probe_duration_seconds(clip_path: Path) -> float:
    """
    Get clip duration using ffprobe.

    Parameters:
        clip_path: Path to video file

    Returns:
        Duration seconds as float (0.0 on failure)
    """
    # Why: We need duration for planning without decoding the whole clip in Python.
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(clip_path),
    ]
    try:
        out = subprocess.check_output(cmd, text=True).strip()
        return float(out) if out else 0.0
    except Exception:
        return 0.0


def _parse_json_strict(text: str) -> Dict[str, Any]:
    """
    Parse JSON from a model response.

    Parameters:
        text: Raw response

    Returns:
        Dict parsed from JSON
    """
    # Why: Vision prompt requires JSON-only; fail fast if the model violates this contract.
    return json.loads(text)


def _vision_json_with_retry(ollama_client: Any, image_path: str) -> Dict[str, Any]:
    """
    Call vision model and enforce JSON-only output with one retry.

    Parameters:
        ollama_client: Dependency-injected client with .vision(image_path, prompt)
        image_path: JPG path

    Returns:
        Parsed JSON dict
    """
    # Why: Models occasionally add prose; retry once with stricter instruction.
    raw = ollama_client.vision(image_path=image_path, prompt=VISION_PROMPT)
    try:
        return _parse_json_strict(raw)
    except json.JSONDecodeError:
        raw2 = ollama_client.vision(
            image_path=image_path,
            prompt=VISION_PROMPT + "\nReturn ONLY valid JSON with no other text.",
        )
        return _parse_json_strict(raw2)


def _normalize_frame_analysis(image_path: str, obj: Dict[str, Any]) -> FrameAnalysis:
    """
    Convert a model JSON response into a validated FrameAnalysis.

    Parameters:
        image_path: Source image path
        obj: JSON object

    Returns:
        FrameAnalysis
    """
    # Why: Director needs consistent fields; normalize model outputs defensively.
    scene_type = str(obj.get("scene_type", "other"))
    subject = str(obj.get("subject", "")).strip() or "unknown"
    best_for = str(obj.get("best_for", "background"))
    motion_level = str(obj.get("motion_level", "static"))
    keep = bool(obj.get("keep", True))
    reason = str(obj.get("reason", "")).strip() or "no reason provided"
    try:
        visual_energy = int(obj.get("visual_energy", 5))
    except (TypeError, ValueError):
        visual_energy = 5
    visual_energy = max(1, min(10, visual_energy))

    return FrameAnalysis(
        image_path=image_path,
        scene_type=scene_type,
        subject=subject,
        visual_energy=visual_energy,
        motion_level=motion_level,
        best_for=best_for,
        keep=keep,
        reason=reason,
    )


def analyze_footage(
    footage_dir: Path,
    temp_dir: Path,
    ollama_client: Any,
) -> List[Dict[str, Any]]:
    """
    Analyze all .mp4 clips inside a footage directory.

    Parameters:
        footage_dir: Directory containing .mp4 clips
        temp_dir: Temp directory root (frames are written here)
        ollama_client: Dependency-injected client with .vision()

    Returns:
        List of ClipMetadata-like dicts suitable for prompting the Director.
    """
    # Why: Keep the output JSON-serializable for direct injection into director prompts.
    if not footage_dir.exists():
        raise FileNotFoundError(f"Footage directory not found: {footage_dir}")
    temp_frames_root = temp_dir / "frames"
    temp_frames_root.mkdir(parents=True, exist_ok=True)

    clips = sorted([p for p in footage_dir.glob("*.mp4") if p.is_file()])
    if not clips:
        raise ValueError(f"No .mp4 clips found in: {footage_dir}")

    results: List[Dict[str, Any]] = []
    for clip_path in clips:
        clip_stem = clip_path.stem
        clip_temp_dir = temp_frames_root / clip_stem
        clip_temp_dir.mkdir(parents=True, exist_ok=True)

        out_pattern = clip_temp_dir / f"{clip_stem}_frame_%04d.jpg"
        _run_ffmpeg_extract_frames(clip_path, out_pattern)

        frame_paths = sorted(clip_temp_dir.glob(f"{clip_stem}_frame_*.jpg"))
        if not frame_paths:
            log.warning("no_frames_extracted", clip=str(clip_path))
            continue

        duration = _probe_duration_seconds(clip_path)

        frame_analyses: List[FrameAnalysis] = []
        for fp in frame_paths:
            obj = _vision_json_with_retry(ollama_client, str(fp))
            frame_analyses.append(_normalize_frame_analysis(str(fp), obj))

        avg_energy = sum(f.visual_energy for f in frame_analyses) / max(1, len(frame_analyses))
        keep_ratio = sum(1 for f in frame_analyses if f.keep) / max(1, len(frame_analyses))

        # Dominant scene_type by frequency (ties break by higher avg energy).
        counts: Dict[str, List[int]] = {}
        for fa in frame_analyses:
            counts.setdefault(fa.scene_type, []).append(fa.visual_energy)
        dominant_scene_type = sorted(
            counts.items(),
            key=lambda kv: (len(kv[1]), sum(kv[1]) / max(1, len(kv[1]))),
            reverse=True,
        )[0][0]

        # Best thumbnail frame: prefer keep==True and best_for hook/action/product, then energy.
        def thumb_key(fa: FrameAnalysis) -> Tuple[int, int]:
            best_for_boost = 0
            if fa.best_for in ("hook", "action_moment", "product_showcase"):
                best_for_boost = 2
            elif fa.best_for in ("transition",):
                best_for_boost = 1
            return (best_for_boost + (1 if fa.keep else 0), fa.visual_energy)

        best_frame = sorted(frame_analyses, key=thumb_key, reverse=True)[0]
        thumbnail_frame = best_frame.image_path if best_frame else None

        clip_meta = ClipMetadata(
            clip_filename=clip_path.name,
            clip_path=str(clip_path),
            duration_seconds=float(duration),
            dominant_scene_type=dominant_scene_type,
            avg_visual_energy=float(avg_energy),
            keep_ratio=float(keep_ratio),
            thumbnail_frame=thumbnail_frame,
            frames=[
                {
                    "image_path": f.image_path,
                    "scene_type": f.scene_type,
                    "subject": f.subject,
                    "visual_energy": f.visual_energy,
                    "motion_level": f.motion_level,
                    "best_for": f.best_for,
                    "keep": f.keep,
                    "reason": f.reason,
                }
                for f in frame_analyses
            ],
        )
        results.append(
            {
                "clip_filename": clip_meta.clip_filename,
                "clip_path": clip_meta.clip_path,
                "duration_seconds": clip_meta.duration_seconds,
                "dominant_scene_type": clip_meta.dominant_scene_type,
                "avg_visual_energy": clip_meta.avg_visual_energy,
                "keep_ratio": clip_meta.keep_ratio,
                "thumbnail_frame": clip_meta.thumbnail_frame,
                "frames": clip_meta.frames,
            }
        )

    if not results:
        raise RuntimeError("Scene analysis produced no usable clip metadata.")
    return results

