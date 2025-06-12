import os

# Set environment variables for vLLM and Gradio behaviour
DEFAULTS = {
    "VLLM_MAX_MODEL_LEN": "4096",
    "VLLM_GPU_MEMORY_UTILIZATION": "0.90",
    "VLLM_DISABLE_LOGGING": "1",
    "VLLM_NO_USAGE_STATS": "1",
    "VLLM_DO_NOT_TRACK": "1",
    "GRADIO_ANALYTICS_ENABLED": "False",
}
for key, val in DEFAULTS.items():
    os.environ.setdefault(key, val)

from vllm.engine.async_llm_engine import AsyncLLMEngine, AsyncEngineArgs

# Patch AsyncLLMEngine.from_engine_args to apply environment defaults
_orig_from_engine_args = AsyncLLMEngine.from_engine_args.__func__  # type: ignore

def _patched_from_engine_args(cls, engine_args: AsyncEngineArgs, *args, **kwargs):
    max_len = os.getenv("VLLM_MAX_MODEL_LEN")
    if max_len:
        try:
            engine_args.max_model_len = int(max_len)
        except ValueError:
            pass
    gpu_util = os.getenv("VLLM_GPU_MEMORY_UTILIZATION")
    if gpu_util:
        try:
            engine_args.gpu_memory_utilization = float(gpu_util)
        except ValueError:
            pass
    return _orig_from_engine_args(cls, engine_args, *args, **kwargs)

AsyncLLMEngine.from_engine_args = classmethod(_patched_from_engine_args)

# Import the existing Gradio app and launch it
import gradio_app


def main() -> None:
    """Launch the Gradio app with patched vLLM settings."""
    gradio_app.demo.launch(server_port=18188)


if __name__ == "__main__":
    main()
