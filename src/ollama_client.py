"""
Unified wrapper for all Ollama model calls.

Why: Keep all LLM/VLM traffic isolated behind one dependency-injected client so the
rest of the pipeline (audio/video/scene logic) stays testable and deterministic.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import structlog

from src.config import (
    MODEL_CREATIVE,
    MODEL_DIRECTOR,
    MODEL_VISION,
    OLLAMA_HOST,
    OLLAMA_TIMEOUT,
)


log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class OllamaModels:
    """Model names used by the pipeline."""

    director: str = MODEL_DIRECTOR
    creative: str = MODEL_CREATIVE
    vision: str = MODEL_VISION


class OllamaClient:
    """
    A strict JSON-first wrapper around an Ollama server.

    Notes:
    - Why: The edit pipeline depends on valid JSON responses; this class enforces that.
    - Timeout: Defaults to 300s because very large cloud-backed models can be slow.
    """

    def __init__(
        self,
        host: str = OLLAMA_HOST,
        timeout_seconds: int = OLLAMA_TIMEOUT,
        models: Optional[OllamaModels] = None,
    ) -> None:
        """
        Initialize the client.

        Parameters:
            host: Ollama base URL, typically http://localhost:11434
            timeout_seconds: Request timeout for all calls
            models: Optional model-name bundle

        Returns:
            None
        """
        # Why: Import lazily so basic tools like `python -m py_compile` don't require deps installed.
        import ollama  # type: ignore

        self._ollama = ollama
        self._client = ollama.Client(host=host, timeout=timeout_seconds)
        self._timeout = timeout_seconds
        self.models = models or OllamaModels()

    def healthcheck(self) -> Dict[str, Any]:
        """
        Check that the Ollama server is reachable and list available tags.

        Returns:
            Dict containing server tag information.
        """
        # Why: Provide a fast preflight so UI can error early with a friendly message.
        return self._client.list()

    def chat_json(self, model: str, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Call an Ollama chat model and parse a JSON object from the response.

        Retry logic:
        - On JSONDecodeError: append instruction to return ONLY valid JSON and retry once.
        - On second failure: raise ValueError with original response for debugging.

        Parameters:
            model: Ollama model name
            messages: Chat messages list

        Returns:
            Parsed JSON dict
        """
        # Why: Many models sometimes wrap JSON in prose; we enforce strict machine output.
        original = None
        for attempt in range(2):
            resp = self._client.chat(model=model, messages=messages)
            content = (resp.get("message") or {}).get("content", "")
            if original is None:
                original = content
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                if attempt == 0:
                    messages = list(messages) + [
                        {
                            "role": "user",
                            "content": "Return ONLY valid JSON with no other text.",
                        }
                    ]
                    continue
                raise ValueError(
                    "Ollama returned invalid JSON after retry. Raw response:\n"
                    + (original or content)
                )
        raise RuntimeError("Unreachable")

    def vision(self, image_path: str, prompt: str) -> str:
        """
        Send an image + prompt to a vision-capable model.

        Parameters:
            image_path: Path to image file on disk
            prompt: Instruction prompt

        Returns:
            Model text output (not parsed)
        """
        # Why: Ollama's vision API accepts images on the message object; we keep parsing separate.
        resp = self._client.chat(
            model=self.models.vision,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_path],
                }
            ],
        )
        return (resp.get("message") or {}).get("content", "")

    def director_decision(
        self,
        script: str,
        audio_data: Dict[str, Any],
        clip_metadata: List[Dict[str, Any]],
        system_prompt: str,
    ) -> Dict[str, Any]:
        """
        Ask the Director model for a full edit plan JSON.

        Parameters:
            script: Narration script text
            audio_data: Output from audio_gen (duration, silences, word timings)
            clip_metadata: Output from scene_analyzer
            system_prompt: Director system prompt defining rules and JSON schema

        Returns:
            Edit plan JSON dict
        """
        # Why: Keep all prompt assembly in one place to simplify debugging and iteration.
        user_payload = {
            "script": script,
            "audio": audio_data,
            "clips": clip_metadata,
        }
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": "Plan the edit. Use ONLY the provided clips. Return ONLY valid JSON.\n"
                + json.dumps(user_payload),
            },
        ]
        return self.chat_json(self.models.director, messages)

    def write_script(
        self,
        clip_metadata: List[Dict[str, Any]],
        style: str = "Outdoorsy",
        platform: str = "TikTok",
        max_length_seconds: float = 60.0,
    ) -> str:
        """
        Ask the Creative model to write a narration script from footage analysis.

        Called after scene_analyzer.analyze_footage() so the model has structured
        per-frame descriptions to work from instead of raw video.

        Parameters:
            clip_metadata: Output from scene_analyzer.analyze_footage()
            style: Content style label (Outdoorsy|Hype|Calm)
            platform: Target platform (TikTok|Reels|Shorts)
            max_length_seconds: Target length — determines approximate word count

        Returns:
            Plain narration text (not JSON) ready to feed directly into Piper TTS.
        """
        # Why: ~150 WPM is a comfortable narration pace for short-form video.
        target_words = int(max_length_seconds * 2.5)

        # Compact summary so the prompt isn't bloated with full frame dicts.
        footage_summary = []
        for clip in clip_metadata:
            footage_summary.append({
                "clip": clip["clip_filename"],
                "duration_s": round(clip["duration_seconds"], 1),
                "dominant_scene": clip["dominant_scene_type"],
                "avg_energy": round(clip["avg_visual_energy"], 1),
                "scenes": [
                    {
                        "scene_type": f["scene_type"],
                        "subject": f["subject"],
                        "energy": f["visual_energy"],
                        "best_for": f["best_for"],
                    }
                    for f in clip.get("frames", [])
                ],
            })

        system = (
            "You are a scriptwriter for viral fishing short-form video for Shorette's Bait and Tackle.\n"
            f"Style: {style.lower()}. Platform: {platform}. Target narration: ~{target_words} words.\n"
            "\nScript rules:\n"
            "- First sentence must be a hook (exciting question, bold claim, or dramatic moment)\n"
            "- Reference actual visual moments from the footage (catches, lures, casts, water)\n"
            "- Use natural, conversational fishing language — not corporate, not cringe\n"
            "- Match energy to visual_energy scores (high-energy scenes = punchy short sentences)\n"
            "- End with a call to action: like, follow, or visit Shorette's\n"
            f"- Keep word count close to {target_words} words for a {max_length_seconds:.0f}s video\n"
            "- NO hashtags in the script. NO stage directions. NO headers. No explanations.\n"
            "Return ONLY the narration script text."
        )

        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": (
                    "The vision model analyzed this footage. "
                    "Write a narration script based on what it saw:\n\n"
                    + json.dumps(footage_summary, indent=2)
                ),
            },
        ]
        # Why: Script output is plain prose — we use chat() directly, not chat_json().
        resp = self._client.chat(model=self.models.creative, messages=messages)
        return (resp.get("message") or {}).get("content", "").strip()

    def write_captions(
        self,
        script: str,
        word_timings: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Ask the Creative model for a caption plan JSON (segments + optional hashtags).

        Parameters:
            script: Narration script
            word_timings: Word-level timing list

        Returns:
            Caption plan JSON dict
        """
        # Why: Captioning style/phrasing is a creative task best handled by a separate model.
        system = (
            "You write captions for viral fishing short-form videos for Shorette's Bait and Tackle.\n"
            "Style: outdoorsy, clean, trustworthy, high-retention. No cringe. No excessive emojis.\n"
            "Return ONLY valid JSON:\n"
            "{\n"
            "  \"style\": \"bold_impact|clean_lower_third\",\n"
            "  \"segments\": [\n"
            "    {\"start\": float, \"end\": float, \"text\": str}\n"
            "  ],\n"
            "  \"hook_text\": str,\n"
            "  \"hashtags\": [str]\n"
            "}\n"
        )
        user_payload = {"script": script, "word_timings": word_timings}
        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": "Create readable on-screen captions aligned to timings. "
                "Use 2-6 words per segment. Return ONLY JSON.\n"
                + json.dumps(user_payload),
            },
        ]
        return self.chat_json(self.models.creative, messages)

