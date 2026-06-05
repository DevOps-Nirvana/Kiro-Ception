"""Memory management utilities for conversation indexing."""

import hashlib
import logging
import os
import platform
import subprocess

from .config import (
    BYTES_PER_MESSAGE,
    DEFAULT_MEMORY_FRACTION,
    MEMORY_LIMIT_DISABLED_ENV,
    MEMORY_LIMIT_ENV,
    get_config,
)
from .models import SessionInfo

logger = logging.getLogger(__name__)


def get_physical_memory() -> int:
    """Get physical memory in bytes."""
    system = platform.system()
    if system == "Linux":
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) * 1024
        except (OSError, ValueError, IndexError):
            pass
    elif system == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return int(result.stdout.strip())
        except (subprocess.SubprocessError, ValueError):
            pass
    return 0


def get_memory_limit() -> int:
    """Get memory limit in bytes for the index."""
    if os.environ.get(MEMORY_LIMIT_DISABLED_ENV):
        return 0

    override = os.environ.get(MEMORY_LIMIT_ENV)
    if override:
        try:
            return int(override) * 1024 * 1024
        except ValueError:
            logger.warning(f"Invalid {MEMORY_LIMIT_ENV} value: {override}")

    config = get_config()
    if config.memory.limit_mb:
        return config.memory.limit_mb * 1024 * 1024

    physical = get_physical_memory()
    fraction = config.memory.fraction or DEFAULT_MEMORY_FRACTION
    return int(physical * fraction) if physical else 0


def select_sessions_within_limit(
    sessions: list[SessionInfo], memory_limit_bytes: int
) -> tuple[list[SessionInfo], list[SessionInfo]]:
    """Select newest sessions that fit within memory limit."""
    if memory_limit_bytes <= 0:
        return sessions, []

    sorted_sessions = sorted(sessions, key=lambda s: s.timestamp_fallback, reverse=True)
    selected, excluded = [], []
    current_bytes = 0

    # Estimate ~10 messages per session if message_count not set
    for session in sorted_sessions:
        msg_count = session.message_count if session.message_count > 0 else 10
        estimated = msg_count * BYTES_PER_MESSAGE
        if current_bytes + estimated <= memory_limit_bytes:
            selected.append(session)
            current_bytes += estimated
        else:
            excluded.append(session)

    return selected, excluded
