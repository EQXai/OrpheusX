import logging
import threading
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "orpheus.log"


def get_logger(name: str = "orpheus") -> logging.Logger:
    """Return a logger writing to both console and file."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(fmt)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


class UILogHandler(logging.Handler):
    """Capture log records for UI streaming."""

    def __init__(self) -> None:
        super().__init__()
        self._buffer: list[str] = []
        self._lock = threading.Lock()
        self.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - small
        msg = self.format(record)
        with self._lock:
            self._buffer.append(msg)

    def pop(self) -> str:
        with self._lock:
            lines = "\n".join(self._buffer)
            self._buffer.clear()
        return lines
