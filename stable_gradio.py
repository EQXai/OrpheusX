#!/usr/bin/env python3
"""Minimal Gradio app designed to avoid ``Connection errored out`` issues.

This example demonstrates how to run long tasks in a background thread while
keeping Gradio responsive.  It also checks for port conflicts and exposes
options to set ``share``, ``server_name`` and ``server_port``.

Replace ``long_task`` with your WhisperX transcription or any heavy process.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import socket
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import gradio as gr


logger = logging.getLogger("stable_gradio")
logging.basicConfig(level=logging.INFO)

# Thread pool used to run blocking tasks without freezing Gradio
EXECUTOR = ThreadPoolExecutor(max_workers=1)


def long_task(audio_file: str) -> str:
    """Example placeholder for a heavy operation.

    Replace this with the actual WhisperX call or any CPU/GPU intensive task.
    """
    logger.info("Starting long task for %%s", audio_file)
    import time

    time.sleep(5)  # simulate heavy work
    logger.info("Finished long task for %%s", audio_file)
    return f"Processed {Path(audio_file).name}"


async def long_task_async(audio_file: str) -> str:
    """Run ``long_task`` in a background thread."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(EXECUTOR, long_task, audio_file)


def check_port_available(port: int, host: str = "127.0.0.1") -> bool:
    """Return True if ``host:port`` can be bound."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def build_interface() -> gr.Blocks:
    with gr.Blocks() as demo:
        gr.Markdown("# Stable Gradio Demo")
        audio_in = gr.Audio(type="filepath", label="Input audio")
        output_box = gr.Textbox(label="Result")
        run_btn = gr.Button("Run")

        run_btn.click(long_task_async, inputs=audio_in, outputs=output_box)
    # queue() keeps the connection alive for long jobs
    demo.queue(concurrency_count=1)
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the stable Gradio server")
    parser.add_argument("--share", action="store_true", help="Use gradio share mode")
    parser.add_argument("--server-name", default="127.0.0.1", help="Server name or IP")
    parser.add_argument("--server-port", type=int, default=7860, help="Server port")
    args = parser.parse_args()

    if not check_port_available(args.server_port, args.server_name):
        raise SystemExit(f"Port {args.server_port} is already in use")

    demo = build_interface()

    logger.info("Launching on %s:%d", args.server_name, args.server_port)
    demo.launch(
        share=args.share,
        server_name=args.server_name,
        server_port=args.server_port,
        show_error=True,
    )


if __name__ == "__main__":
    main()
