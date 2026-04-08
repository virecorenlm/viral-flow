"""
Gradio UI entrypoint for Viral-Flow.

Why: Provide a drag-and-drop local interface that stays responsive while the
render pipeline runs in a background thread.
"""

from __future__ import annotations

import sys
import json
import queue
import shutil
import threading
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import structlog

# Why: Allow `python src/ui_main.py` to import `src.*` by ensuring project root is on sys.path.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.audio_gen import generate_narration_audio
from src.config import FOOTAGE_DIR, INPUT_DIR, OUTPUT_DIR, TEMP_DIR
from src.ollama_client import OllamaClient
from src.scene_analyzer import analyze_footage
from src.script_writer import generate_script_from_footage
from src.video_logic import render_from_plan


log = structlog.get_logger(__name__)


CSS = """
body, .gradio-container {
  background: #0a0a10 !important;
}
.vf-title {
  font-size: 22px;
  font-weight: 800;
  letter-spacing: 0.5px;
}
.vf-accent {
  color: #00e5ff;
}
.vf-status {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
  white-space: pre-wrap;
}
"""


@dataclass
class JobState:
    """State for a single render job."""

    thread: Optional[threading.Thread] = None
    q: "queue.Queue[Dict[str, Any]]" = field(default_factory=queue.Queue)
    done: bool = False
    failed: bool = False
    output_path: Optional[str] = None
    director_notes: str = ""
    last_status: str = "Idle."
    last_error: str = ""


def _ensure_dirs() -> None:
    """
    Ensure required directories exist.

    Returns:
        None
    """
    # Why: First-run UX should not require manual folder creation.
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    FOOTAGE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)


def _clean_temp(temp_dir: Path) -> None:
    """
    Remove intermediate files in temp directory.

    Parameters:
        temp_dir: Temp directory root

    Returns:
        None
    """
    # Why: Temp files can grow quickly during video work; auto-clean keeps disk healthy.
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
    temp_dir.mkdir(parents=True, exist_ok=True)


def _copy_inputs(
    script_file: Any,
    footage_files: List[Any],
) -> Tuple[Optional[Path], Path]:
    """
    Copy uploaded footage (required) and optional script into project input folders.

    Parameters:
        script_file: Gradio File object, or None when auto-generating from footage
        footage_files: List of Gradio File objects (.mp4)

    Returns:
        (script_path or None, footage_dir)
        script_path is None when no script was uploaded — caller must auto-generate.
    """
    # Why: Keep the pipeline file-path based and predictable, independent of upload temp paths.
    _ensure_dirs()

    script_dst: Optional[Path] = None
    if script_file is not None:
        script_src = Path(getattr(script_file, "name", str(script_file)))
        if script_src.exists():
            script_dst = INPUT_DIR / "script.txt"
            shutil.copyfile(script_src, script_dst)

    # Clear old footage then copy new clips.
    for old in FOOTAGE_DIR.glob("*.mp4"):
        try:
            old.unlink()
        except Exception:
            pass

    if not footage_files:
        raise ValueError("No footage files uploaded.")

    for f in footage_files:
        src = Path(getattr(f, "name", str(f)))
        if not src.exists():
            continue
        if src.suffix.lower() != ".mp4":
            continue
        shutil.copyfile(src, FOOTAGE_DIR / src.name)

    if not list(FOOTAGE_DIR.glob("*.mp4")):
        raise ValueError("No .mp4 footage files detected after copy.")

    return script_dst, FOOTAGE_DIR


def _format_status(steps: Dict[str, str]) -> str:
    """
    Format pipeline step statuses into a UI-friendly block.

    Parameters:
        steps: Mapping of step name -> status marker

    Returns:
        Status block string
    """
    # Why: Keep the UI readable and predictable during long renders.
    order = [
        "Generating script",
        "Generating voiceover",
        "Analyzing footage",
        "AI Director planning cuts",
        "Rendering video",
        "Burning captions",
        "Cleanup",
    ]
    lines = ["PIPELINE STATUS:"]
    for k in order:
        lines.append(f"{steps.get(k, '○')} {k}")
    return "\n".join(lines)


def _pipeline_worker(
    job: JobState,
    script_file: Any,
    footage_files: List[Any],
    platform: str,
    style: str,
    max_length: str,
) -> None:
    """
    Background thread worker that runs the full pipeline.

    Parameters:
        job: JobState (contains status queue)
        script_file: Uploaded script file
        footage_files: Uploaded mp4 files
        platform: Platform choice label
        style: Style choice label
        max_length: Max length choice label

    Returns:
        None
    """
    steps = {k: "○" for k in [
        "Generating script",
        "Generating voiceover",
        "Analyzing footage",
        "AI Director planning cuts",
        "Rendering video",
        "Burning captions",
        "Cleanup",
    ]}

    def push():
        job.q.put({"type": "status", "text": _format_status(steps)})

    try:
        _clean_temp(TEMP_DIR)

        # Max length mapping (needed early for script word-count targeting).
        max_len: Optional[float] = None
        if max_length == "30s":
            max_len = 30.0
        elif max_length == "60s":
            max_len = 60.0
        elif max_length == "90s":
            max_len = 90.0

        script_path, _ = _copy_inputs(script_file, footage_files)
        client = OllamaClient()
        clip_metadata = None  # May be pre-populated when auto-generating the script.

        if script_path is None:
            # ── AUTO-GENERATE MODE ───────────────────────────────────────────────
            # Vision model pre-watches footage → creative model writes narration.
            # analyze_footage() runs inside generate_script_from_footage() and its
            # clip_metadata is returned so we DON'T call the vision model twice.
            steps["Generating script"] = "⟳"
            push()
            script_text, clip_metadata = generate_script_from_footage(
                footage_dir=FOOTAGE_DIR,
                temp_dir=TEMP_DIR,
                ollama_client=client,
                style=style,
                platform=platform,
                max_length_seconds=max_len or 60.0,
                save_to=INPUT_DIR / "script.txt",
            )
            steps["Generating script"] = "✓"
            steps["Analyzing footage"] = "✓"  # Already done inside script generation.
            push()
        else:
            # ── USER-PROVIDED SCRIPT MODE ────────────────────────────────────────
            script_text = script_path.read_text(encoding="utf-8").strip()
            if not script_text:
                raise ValueError("Script is empty after upload.")
            steps["Generating script"] = "✓"
            push()

        steps["Generating voiceover"] = "⟳"
        push()
        audio_data = generate_narration_audio(
            script_text=script_text,
            output_wav_path=TEMP_DIR / "narration.wav",
        )
        steps["Generating voiceover"] = "✓"
        push()

        if clip_metadata is None:
            # Vision analysis hasn't run yet (user provided a script manually).
            steps["Analyzing footage"] = "⟳"
            push()
            clip_metadata = analyze_footage(
                footage_dir=FOOTAGE_DIR,
                temp_dir=TEMP_DIR,
                ollama_client=client,
            )
            steps["Analyzing footage"] = "✓"
            push()

        steps["AI Director planning cuts"] = "⟳"
        push()

        # Render does director+captions internally; we set statuses around it.
        steps["AI Director planning cuts"] = "✓"
        steps["Rendering video"] = "⟳"
        push()
        result = render_from_plan(
            script_text=script_text,
            audio_data=audio_data,
            clip_metadata=clip_metadata,
            ollama_client=client,
            max_length_seconds=max_len,
            output_dir=OUTPUT_DIR,
            footage_dir=FOOTAGE_DIR,
            temp_dir=TEMP_DIR,
        )
        steps["Rendering video"] = "✓"
        steps["Burning captions"] = "✓"
        push()

        steps["Cleanup"] = "⟳"
        push()
        _clean_temp(TEMP_DIR)
        steps["Cleanup"] = "✓"
        push()

        job.output_path = result.output_path
        job.director_notes = result.director_notes
        job.done = True
        job.q.put({"type": "done", "output_path": result.output_path, "notes": result.director_notes})
    except Exception as e:
        job.failed = True
        job.last_error = f"{e}\n\nCheck console for full traceback."
        tb = traceback.format_exc()
        log.error("pipeline_failed", error=str(e), traceback=tb)
        job.q.put({"type": "error", "message": job.last_error})


def start_job(
    script_file: Any,
    footage_files: List[Any],
    platform: str,
    style: str,
    max_length: str,
    job: JobState,
) -> Tuple[str, JobState, Optional[str], Optional[str], str]:
    """
    Start a background render job.

    Parameters:
        script_file: Uploaded script file
        footage_files: List of uploaded footage files
        platform: Platform selection
        style: Style selection
        max_length: Length selection
        job: Existing JobState from UI

    Returns:
        (status_text, job_state, video_path, download_path, director_notes)
    """
    # Why: Gradio handlers should return immediately; the thread does the heavy work.
    if job.thread and job.thread.is_alive():
        return job.last_status, job, job.output_path, job.output_path, job.director_notes

    new_job = JobState()
    new_job.last_status = "Starting pipeline..."
    th = threading.Thread(
        target=_pipeline_worker,
        args=(new_job, script_file, footage_files, platform, style, max_length),
        daemon=True,
    )
    new_job.thread = th
    th.start()
    return new_job.last_status, new_job, None, None, ""


def poll_job(job: JobState) -> Tuple[str, Optional[str], Optional[str], str, str]:
    """
    Poll the job queue for updates.

    Parameters:
        job: JobState

    Returns:
        (status_text, video_path, download_path, director_notes, error_text)
    """
    # Why: Timer-driven polling keeps UI responsive without blocking.
    status = job.last_status
    notes = job.director_notes
    err = job.last_error if job.failed else ""
    vid = job.output_path
    dl = job.output_path

    drained = False
    while not job.q.empty():
        drained = True
        msg = job.q.get_nowait()
        if msg.get("type") == "status":
            status = str(msg.get("text", status))
        elif msg.get("type") == "done":
            vid = str(msg.get("output_path"))
            dl = vid
            notes = str(msg.get("notes", ""))
            status = status + "\n\n✓ Done."
        elif msg.get("type") == "error":
            err = str(msg.get("message", "Unknown error"))
            status = status + "\n\n✗ Failed."

    if drained:
        job.last_status = status
        job.director_notes = notes
        job.last_error = err
        job.output_path = vid

    return status, vid, dl, notes, err


def build_ui() -> Any:
    """
    Build and return the Gradio app.

    Returns:
        Gradio Blocks app
    """
    # Why: Keep top-level module import lightweight; Gradio is only needed at runtime.
    import gradio as gr  # type: ignore

    _ensure_dirs()

    theme = gr.themes.Base()

    with gr.Blocks(theme=theme, css=CSS, title="Viral-Flow") as demo:
        job_state = gr.State(JobState())

        gr.HTML(
            "<div class='vf-title'>🎣 <span class='vf-accent'>VIRAL-FLOW</span> "
            "│ Shorette's Bait &amp; Tackle</div>"
        )

        with gr.Row():
            script_file = gr.File(
                label="DROP SCRIPT.TXT (optional — leave blank to auto-write from footage)",
                file_types=[".txt"],
            )
            footage_files = gr.Files(label="DROP FOOTAGE FILES HERE (.mp4)", file_types=[".mp4"])

        with gr.Row():
            platform = gr.Radio(["TikTok", "Reels", "Shorts"], value="TikTok", label="Platform")
            style = gr.Radio(["Outdoorsy", "Hype", "Calm"], value="Outdoorsy", label="Style")
            max_length = gr.Radio(["30s", "60s", "90s"], value="60s", label="Max Length")

        generate_btn = gr.Button("▶ GENERATE VIDEO", variant="primary")

        status_box = gr.Textbox(
            label="PIPELINE STATUS",
            value="PIPELINE STATUS:\n○ Generating script\n○ Generating voiceover\n○ Analyzing footage\n○ AI Director planning cuts\n○ Rendering video\n○ Burning captions\n○ Cleanup",
            lines=10,
            elem_classes=["vf-status"],
        )

        error_box = gr.Textbox(label="ERROR", value="", lines=4, visible=True)

        video_out = gr.Video(label="VIDEO PREVIEW PLAYER", interactive=False)
        download_out = gr.File(label="⬇ DOWNLOAD OUTPUT")

        notes_box = gr.Textbox(label="AI DIRECTOR NOTES", value="", lines=6)

        def _start(script_f, footage_fs, plat, sty, mx, job):
            return start_job(script_f, footage_fs, plat, sty, mx, job)

        generate_btn.click(
            _start,
            inputs=[script_file, footage_files, platform, style, max_length, job_state],
            outputs=[status_box, job_state, video_out, download_out, notes_box],
        )

        timer = gr.Timer(0.8)

        def _poll(job):
            status, vid, dl, notes, err = poll_job(job)
            return status, vid, dl, notes, err

        timer.tick(
            _poll,
            inputs=[job_state],
            outputs=[status_box, video_out, download_out, notes_box, error_box],
        )

    return demo


if __name__ == "__main__":
    # Why: Direct run entrypoint requested by setup.sh and README.
    app = build_ui()
    app.launch(server_name="127.0.0.1", server_port=7860)

