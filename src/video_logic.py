"""
AI Director + MoviePy rendering pipeline.

Why: This module is the core editor that turns narration + clip metadata into a
viral-ready 9:16 vertical video for TikTok/Reels, with captions and retention pacing.
"""

from __future__ import annotations

import json
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import structlog

from src.config import (
    BRAND_STYLE,
    FOOTAGE_DIR,
    OUTPUT_CRF,
    OUTPUT_DIR,
    OUTPUT_FPS,
    OUTPUT_HEIGHT,
    OUTPUT_WIDTH,
    PLATFORM_TARGET,
    TEMP_DIR,
)


log = structlog.get_logger(__name__)


DIRECTOR_SYSTEM_PROMPT = f"""
You are a viral fishing content video editor for a brand called Shorette's Bait and Tackle.
Your editing style: outdoorsy, clean, trustworthy. Fast-paced but never chaotic.
Think: Outdoor fishing YouTube meets viral TikTok. High retention. Never boring.

Brand style: {BRAND_STYLE}
Platform target: {PLATFORM_TARGET}

Rules:
- First 3 seconds = the HOOK. Use the most exciting visual (big catch, lure splash, action).
- Cut on natural audio pauses whenever possible (silence_gaps provided).
- Match visual energy to narration energy — calm narration = wider shot, excited = tight/closeup.
- Every 4-6 seconds something must change visually (cut, zoom, or transition).
- Use Ken Burns (slow zoom) on static shots to maintain motion.
- Lure/product closeups: hold for 2-3 seconds minimum so viewer can see the product.
- End with a clear call to action moment (talking head or text overlay if available).

Given: script, audio duration, silence gaps, and available clips with scene metadata.
Return ONLY valid JSON:
{{
  "total_duration": float,
  "hook_clip": {{"clip_filename": str, "start": float, "end": float, "reason": str}},
  "timeline": [
    {{
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
      "reason": str
    }}
  ],
  "caption_style": "bold_impact|clean_lower_third",
  "thumbnail_clip": str,
  "estimated_retention_score": 1-10
}}
"""


@dataclass(frozen=True)
class RenderResult:
    """Output artifacts produced by render."""

    output_path: str
    edit_plan: Dict[str, Any]
    caption_plan: Dict[str, Any]
    director_notes: str


def _require_ffmpeg() -> None:
    """
    Ensure FFmpeg is on PATH.

    Returns:
        None
    """
    # Why: MoviePy depends on ffmpeg; fail early with a precise message.
    if shutil.which("ffmpeg") is None:
        raise EnvironmentError("FFmpeg not found on PATH. Install ffmpeg system-wide.")


def _now_slug() -> str:
    """
    Generate a timestamp-based slug for filenames.

    Returns:
        A compact string suitable for filenames.
    """
    # Why: Avoid collisions when rendering multiple jobs.
    return time.strftime("%Y%m%d_%H%M%S")


def _safe_float(x: Any, default: float) -> float:
    """
    Convert value to float with default.

    Parameters:
        x: Input value
        default: Default if conversion fails

    Returns:
        Float value
    """
    # Why: Director JSON can drift; this prevents render crashes from minor schema issues.
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _clamp(val: float, lo: float, hi: float) -> float:
    """
    Clamp a numeric value between bounds.

    Parameters:
        val: Value
        lo: Min
        hi: Max

    Returns:
        Clamped value
    """
    return max(lo, min(hi, val))


def _to_vertical_crop(clip: Any) -> Any:
    """
    Convert an arbitrary clip to 1080x1920 by scaling to height and center-cropping.

    Parameters:
        clip: MoviePy clip

    Returns:
        MoviePy clip sized to vertical output
    """
    # Why: Fishing footage is often 16:9; center-crop is the cleanest default.
    from moviepy.video.fx.all import crop  # type: ignore

    scaled = clip.resize(height=OUTPUT_HEIGHT)
    return crop(scaled, x_center=scaled.w / 2, y_center=scaled.h / 2, width=OUTPUT_WIDTH, height=OUTPUT_HEIGHT)


def _to_vertical_blur_pad(clip: Any) -> Any:
    """
    Convert to 1080x1920 by blurring a full-screen background and overlaying a sharp foreground.

    Parameters:
        clip: MoviePy clip

    Returns:
        Composite clip with blurred padding
    """
    # Why: Some shots have important edges; blur-pad preserves the full frame content.
    from moviepy.editor import CompositeVideoClip  # type: ignore
    from moviepy.video.fx.all import crop, gaussian_blur  # type: ignore

    bg = clip.resize(height=OUTPUT_HEIGHT)
    bg = crop(bg, x_center=bg.w / 2, y_center=bg.h / 2, width=OUTPUT_WIDTH, height=OUTPUT_HEIGHT)
    bg = bg.fx(gaussian_blur, sigma=10)

    fg = clip.resize(width=OUTPUT_WIDTH).set_position(("center", "center"))
    return CompositeVideoClip([bg, fg], size=(OUTPUT_WIDTH, OUTPUT_HEIGHT))


def _apply_effect(clip: Any, effect: str) -> Any:
    """
    Apply director-selected motion effects.

    Parameters:
        clip: MoviePy clip
        effect: Effect name

    Returns:
        Transformed clip
    """
    # Why: Subtle motion keeps retention without making outdoors footage feel chaotic.
    if effect == "none":
        return clip

    from moviepy.video.fx.all import crop  # type: ignore

    def zoom_with_center_crop(scale_func):
        z = clip.resize(scale_func)
        return crop(z, x_center=z.w / 2, y_center=z.h / 2, width=OUTPUT_WIDTH, height=OUTPUT_HEIGHT)

    if effect == "ken_burns_in":
        dur = max(0.01, float(clip.duration))
        return zoom_with_center_crop(lambda t: 1.0 + 0.05 * (t / dur))
    if effect == "ken_burns_out":
        dur = max(0.01, float(clip.duration))
        return zoom_with_center_crop(lambda t: 1.05 - 0.05 * (t / dur))
    if effect == "zoom_punch":
        # Quick punch at the beginning of the segment (~0.15s), then settle.
        return zoom_with_center_crop(lambda t: 1.0 + 0.08 * math.exp(-t * 14.0))

    return clip


def _pillow_text_image(
    text: str,
    style: str,
    max_width: int = 980,
) -> Any:
    """
    Render caption text to an RGBA image using Pillow.

    Parameters:
        text: Caption text
        style: Caption style key
        max_width: Max text area width

    Returns:
        Pillow Image (RGBA)
    """
    # Why: MoviePy TextClip often needs ImageMagick; Pillow keeps this fully local & portable.
    from PIL import Image, ImageDraw, ImageFont  # type: ignore

    text = text.strip()
    if not text:
        return Image.new("RGBA", (1, 1), (0, 0, 0, 0))

    if style == "bold_impact":
        font_size = 78
        stroke_width = 6
    else:
        font_size = 62
        stroke_width = 5

    try:
        font = ImageFont.truetype("arialbd.ttf" if style == "bold_impact" else "arial.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    # Rough wrap by words to fit max_width.
    words = text.split()
    lines: List[str] = []
    current = ""
    img_probe = Image.new("RGBA", (max_width, 10), (0, 0, 0, 0))
    draw_probe = ImageDraw.Draw(img_probe)
    for w in words:
        candidate = (current + " " + w).strip()
        bbox = draw_probe.textbbox((0, 0), candidate, font=font, stroke_width=stroke_width)
        if bbox[2] <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = w
    if current:
        lines.append(current)

    # Measure final size.
    line_heights = []
    line_widths = []
    for line in lines:
        bbox = draw_probe.textbbox((0, 0), line, font=font, stroke_width=stroke_width)
        line_widths.append(bbox[2] - bbox[0])
        line_heights.append(bbox[3] - bbox[1])
    w = min(max_width, max(line_widths) if line_widths else 1) + 20
    h = (sum(line_heights) if line_heights else 1) + 20 + (8 * max(0, len(lines) - 1))

    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    y = 10
    for i, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=font, stroke_width=stroke_width)
        lw = bbox[2] - bbox[0]
        x = (w - lw) // 2
        draw.text(
            (x, y),
            line,
            font=font,
            fill=(255, 255, 255, 255),
            stroke_width=stroke_width,
            stroke_fill=(0, 0, 0, 255),
        )
        y += line_heights[i] + 8

    return img


def _caption_clips(caption_plan: Dict[str, Any], total_duration: float) -> List[Any]:
    """
    Build caption overlay clips from a caption plan.

    Parameters:
        caption_plan: JSON dict with segments
        total_duration: Duration of final video

    Returns:
        List of MoviePy clips to composite over the video
    """
    # Why: Captions drive retention on TikTok/Reels; these overlays must be stable and timed.
    from moviepy.editor import ImageClip  # type: ignore
    import numpy as np  # type: ignore

    style = str(caption_plan.get("style", "bold_impact"))
    segments = caption_plan.get("segments") or []
    out = []
    for seg in segments:
        try:
            start = float(seg["start"])
            end = float(seg["end"])
            text = str(seg["text"])
        except Exception:
            continue
        start = _clamp(start, 0.0, total_duration)
        end = _clamp(end, 0.0, total_duration)
        if end <= start:
            continue

        img = _pillow_text_image(text=text, style=style)
        arr = np.array(img)
        ic = (
            ImageClip(arr, ismask=False)
            .set_start(start)
            .set_end(end)
            .set_position(("center", int(OUTPUT_HEIGHT * 0.72)))
        )
        out.append(ic)
    return out


def _fallback_caption_plan(word_timings: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build a simple caption plan if the creative model fails.

    Parameters:
        word_timings: Word timing list

    Returns:
        Caption plan dict
    """
    # Why: Rendering should still complete even if the caption model is unavailable.
    segments = []
    buffer_words: List[Tuple[str, float, float]] = []
    for w in word_timings:
        try:
            word = str(w["word"])
            start = float(w["start"])
            end = float(w["end"])
        except Exception:
            continue
        buffer_words.append((word, start, end))
        if len(buffer_words) >= 5:
            segments.append(
                {
                    "start": buffer_words[0][1],
                    "end": buffer_words[-1][2],
                    "text": " ".join(x[0] for x in buffer_words),
                }
            )
            buffer_words = []
    if buffer_words:
        segments.append(
            {
                "start": buffer_words[0][1],
                "end": buffer_words[-1][2],
                "text": " ".join(x[0] for x in buffer_words),
            }
        )
    return {"style": "bold_impact", "segments": segments, "hook_text": "", "hashtags": []}


def _fallback_edit_plan(audio_data: Dict[str, Any], clip_metadata: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build a minimal edit plan if the director model is unavailable.

    Parameters:
        audio_data: Audio analysis dict
        clip_metadata: Scene metadata list

    Returns:
        Edit plan JSON dict
    """
    # Why: Production-grade behavior means "degrade gracefully" instead of hard failing.
    total = float(audio_data.get("duration_seconds") or 0.0)
    clips_sorted = sorted(
        clip_metadata,
        key=lambda c: float(c.get("avg_visual_energy") or 0.0),
        reverse=True,
    )
    if not clips_sorted:
        raise ValueError("No clips available for fallback edit plan.")
    hook = clips_sorted[0]["clip_filename"]
    timeline = []
    t = 0.0
    order = 1
    for c in clips_sorted:
        dur = float(c.get("duration_seconds") or 0.0) or 3.0
        seg = min(4.5, dur, max(0.0, total - t))
        if seg <= 0:
            break
        timeline.append(
            {
                "order": order,
                "clip_filename": c["clip_filename"],
                "clip_start": 0.0,
                "clip_end": float(seg),
                "audio_start": float(t),
                "audio_end": float(t + seg),
                "framing": "crop_center",
                "effect": "ken_burns_in" if float(c.get("avg_visual_energy") or 5.0) < 6 else "none",
                "transition": "cut",
                "transition_duration": 0.0,
                "reason": "Fallback pacing: quick cuts every ~4-5 seconds.",
            }
        )
        order += 1
        t += seg
        if t >= total:
            break
    return {
        "total_duration": total,
        "hook_clip": {"clip_filename": hook, "start": 0.0, "end": 3.0, "reason": "Fallback: highest energy clip."},
        "timeline": timeline,
        "caption_style": "bold_impact",
        "thumbnail_clip": hook,
        "estimated_retention_score": 7,
    }


def render_from_plan(
    script_text: str,
    audio_data: Dict[str, Any],
    clip_metadata: List[Dict[str, Any]],
    ollama_client: Any,
    max_length_seconds: Optional[float] = None,
    output_dir: Path = OUTPUT_DIR,
    footage_dir: Path = FOOTAGE_DIR,
    temp_dir: Path = TEMP_DIR,
) -> RenderResult:
    """
    End-to-end render: director plan → MoviePy timeline → captions → export.

    Parameters:
        script_text: Script content
        audio_data: Output of audio_gen.generate_narration_audio()
        clip_metadata: Output of scene_analyzer.analyze_footage()
        ollama_client: Dependency-injected client with director_decision/write_captions
        max_length_seconds: Optional hard cap for video duration
        output_dir: Output directory
        footage_dir: Directory containing source clips
        temp_dir: Temp directory root

    Returns:
        RenderResult with output path and model plans.
    """
    _require_ffmpeg()
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    audio_path = Path(str(audio_data.get("audio_path") or ""))
    if not audio_path.exists():
        raise FileNotFoundError(f"Narration audio not found: {audio_path}")

    audio_duration = float(audio_data.get("duration_seconds") or 0.0)
    if audio_duration <= 0:
        raise ValueError("Audio duration is invalid.")

    target_duration = audio_duration
    if max_length_seconds is not None:
        target_duration = min(target_duration, float(max_length_seconds))

    # Director plan.
    try:
        edit_plan = ollama_client.director_decision(
            script=script_text,
            audio_data=audio_data,
            clip_metadata=clip_metadata,
            system_prompt=DIRECTOR_SYSTEM_PROMPT,
        )
    except Exception as e:
        log.exception("director_failed", error=str(e))
        edit_plan = _fallback_edit_plan(audio_data=audio_data, clip_metadata=clip_metadata)

    # Creative captions.
    try:
        caption_plan = ollama_client.write_captions(
            script=script_text,
            word_timings=audio_data.get("word_timings") or [],
        )
    except Exception as e:
        log.exception("captions_failed", error=str(e))
        caption_plan = _fallback_caption_plan(word_timings=audio_data.get("word_timings") or [])

    # MoviePy assembly.
    from moviepy.editor import (  # type: ignore
        AudioFileClip,
        CompositeVideoClip,
        VideoFileClip,
    )

    narration_audio = AudioFileClip(str(audio_path)).subclip(0, target_duration)

    timeline = edit_plan.get("timeline") or []
    if not isinstance(timeline, list) or not timeline:
        raise ValueError("Edit plan timeline is empty.")

    segments = []
    cursor_t = 0.0
    for entry in sorted(timeline, key=lambda x: int(x.get("order", 0))):
        clip_filename = str(entry.get("clip_filename", "")).strip()
        if not clip_filename:
            continue
        src_path = footage_dir / clip_filename
        if not src_path.exists():
            log.warning("missing_clip", clip=clip_filename)
            continue

        clip_start = _safe_float(entry.get("clip_start"), 0.0)
        clip_end = _safe_float(entry.get("clip_end"), 0.0)
        if clip_end <= clip_start:
            continue

        framing = str(entry.get("framing") or "crop_center")
        effect = str(entry.get("effect") or "none")
        transition = str(entry.get("transition") or "cut")
        transition_duration = _safe_float(entry.get("transition_duration"), 0.0)
        transition_duration = _clamp(transition_duration, 0.0, 1.0)

        base = VideoFileClip(str(src_path)).subclip(clip_start, clip_end)

        if framing == "blur_pad":
            seg = _to_vertical_blur_pad(base)
        else:
            seg = _to_vertical_crop(base)

        seg = _apply_effect(seg, effect)

        seg_duration = float(seg.duration)
        if seg_duration <= 0:
            continue

        # Apply transitions via overlaps. For "smash_cut", we just hard cut (optionally could add punch).
        if transition == "fade" and transition_duration > 0:
            seg = seg.crossfadein(transition_duration)
            # Overlap start time to create crossfade.
            start_t = max(0.0, cursor_t - transition_duration)
        else:
            start_t = cursor_t

        seg = seg.set_start(start_t)
        segments.append(seg)

        cursor_t = start_t + seg_duration
        if cursor_t >= target_duration:
            break

    if not segments:
        raise RuntimeError("No segments could be built from the edit plan.")

    base_video = CompositeVideoClip(segments, size=(OUTPUT_WIDTH, OUTPUT_HEIGHT)).set_duration(target_duration)
    base_video = base_video.set_audio(narration_audio)

    caption_overlays = _caption_clips(caption_plan=caption_plan, total_duration=target_duration)
    final = CompositeVideoClip([base_video] + caption_overlays, size=(OUTPUT_WIDTH, OUTPUT_HEIGHT)).set_duration(target_duration)
    final = final.set_audio(narration_audio)

    output_path = output_dir / f"viral_flow_{_now_slug()}.mp4"
    final.write_videofile(
        str(output_path),
        fps=OUTPUT_FPS,
        codec="libx264",
        audio_codec="aac",
        ffmpeg_params=["-crf", str(OUTPUT_CRF), "-movflags", "+faststart"],
        threads=4,
        preset="fast",
    )

    notes = (
        f"Retention estimate: {edit_plan.get('estimated_retention_score', '?')}/10\n"
        f"Hook clip: {((edit_plan.get('hook_clip') or {}).get('clip_filename') or '')}\n"
        f"Caption style: {caption_plan.get('style', edit_plan.get('caption_style', 'bold_impact'))}\n"
    )

    return RenderResult(
        output_path=str(output_path),
        edit_plan=edit_plan,
        caption_plan=caption_plan,
        director_notes=notes,
    )

