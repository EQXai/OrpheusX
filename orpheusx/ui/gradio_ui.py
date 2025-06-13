from __future__ import annotations

import os
import signal
import gradio as gr

from orpheusx import constants as _c
from orpheusx.pipeline.data import list_source_audio, refresh_lists
from orpheusx.pipeline.full import (
    dataset_status_multi,
    run_full_pipeline_batch,
)

# -----------------------------------------------------------------------------
# Task control helpers
# -----------------------------------------------------------------------------

def stop_current() -> str:
    """Signal long-running background tasks to stop."""
    _c.STOP_FLAG = True
    return "Stop signal sent"


def exit_app() -> None:
    """Terminate the whole Gradio process (SIGINT)."""
    os.kill(os.getpid(), signal.SIGINT)

# -----------------------------------------------------------------------------
# Main UI builder
# -----------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    """Return the fully-built Gradio Blocks interface."""
    prompt_files = [""] + list(os.listdir(_c.PROMPT_LIST_DIR)) if _c.PROMPT_LIST_DIR.exists() else [""]

    with gr.Blocks() as demo:
        gr.Markdown("# OrpheusX Gradio Interface")

        refresh_btn = gr.Button("Refresh directories")
        stop_btn = gr.Button("Stop Task")
        exit_btn = gr.Button("Exit")

        with gr.Tabs():
            with gr.Tab("Unified"):
                with gr.Tabs():
                    with gr.Tab("Auto Pipeline"):
                        auto_dataset = gr.Dropdown(
                            choices=list_source_audio(), label="Dataset", multiselect=True
                        )
                        auto_status = gr.Markdown()
                        auto_prompt = gr.Textbox(label="Prompt")
                        auto_batch = gr.Slider(1, 5, step=1, value=1, label="Batch")
                        auto_prompt_file = gr.Dropdown(
                            choices=prompt_files, label="Prompt List"
                        )
                        auto_btn = gr.Button("Run Pipeline")
                        auto_log = gr.Textbox()
                        auto_output = gr.HTML()

                        auto_dataset.change(dataset_status_multi, auto_dataset, auto_status)
                        auto_btn.click(
                            run_full_pipeline_batch,
                            [auto_dataset, auto_prompt, auto_prompt_file, auto_batch],
                            [auto_log, auto_output],
                        )

        # Header buttons
        refresh_btn.click(refresh_lists, None, [auto_dataset, auto_prompt_file])
        stop_btn.click(stop_current, None, None)
        exit_btn.click(exit_app, None, None)

    return demo

__all__ = ["build_ui"] 