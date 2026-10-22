"""Peak utilization of every rate-limit window an account went through.

The usage store keeps only each account's latest reading, so a window's
final percentage is lost the moment it resets. Every accepted reading is
also folded in here, keyed by the window's reset time::

    {"schemaVersion": 1,
     "accounts": {"2": {"five_hour": {"1792560000": 87.0, ...},
                        "seven_day": {...}}}}

Keys are the reset instant rounded to the minute (the API's ``resets_at``
carries jittering sub-second digits). That gives the session history view
exact window boundaries and a peak to divide token counts by. Writes are
best-effort: a lost update only drops one sample of a running maximum.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path

from claude_swap.locking import FileLock
from claude_swap.settings import atomic_write_json

SCHEMA_VERSION = 1
FILENAME = "window_peaks.json"
WINDOW_KEYS = ("five_hour", "seven_day")
# Windows whose reset is older than this are pruned on write.
RETENTION_S = 90 * 86400

_logger = logging.getLogger("claude-swap")


def reset_key(resets_at: object) -> int | None:
    """Epoch seconds of a ``resets_at`` ISO string, rounded to the minute."""
    if not isinstance(resets_at, str) or not resets_at:
        return None
    try:
        ts = datetime.fromisoformat(resets_at.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None
    return int(round(ts / 60.0)) * 60


def _read(path: Path) -> dict:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict) or raw.get("schemaVersion") != SCHEMA_VERSION:
        return {}
    accounts = raw.get("accounts")
    return accounts if isinstance(accounts, dict) else {}


def note_usage(
    cache_dir: Path, readings: dict[str, dict], now: float | None = None
) -> None:
    """Fold ``{account_number: usage_dict}`` readings into the peak file."""
    now = time.time() if now is None else now
    samples: list[tuple[str, str, int, float]] = []
    for num, usage in readings.items():
        if not isinstance(usage, dict):
            continue
        for key in WINDOW_KEYS:
            window = usage.get(key)
            if not isinstance(window, dict):
                continue
            pct = window.get("pct")
            reset = reset_key(window.get("resets_at"))
            if reset is None or isinstance(pct, bool) or not isinstance(pct, (int, float)):
                continue
            samples.append((str(num), key, reset, float(pct)))
    if not samples:
        return
    path = cache_dir / FILENAME
    try:
        with FileLock(cache_dir / f".{FILENAME}.lock"):
            accounts = _read(path)
            for num, key, reset, pct in samples:
                windows = accounts.setdefault(num, {}).setdefault(key, {})
                current = windows.get(str(reset))
                if not isinstance(current, (int, float)) or pct > current:
                    windows[str(reset)] = pct
            cutoff = now - RETENTION_S
            for windows_by_key in accounts.values():
                if not isinstance(windows_by_key, dict):
                    continue
                for key, windows in list(windows_by_key.items()):
                    if isinstance(windows, dict):
                        windows_by_key[key] = {
                            r: p for r, p in windows.items()
                            if r.lstrip("-").isdigit() and int(r) >= cutoff
                        }
            atomic_write_json(path, {"schemaVersion": SCHEMA_VERSION, "accounts": accounts})
    except Exception as e:  # never let history break a usage write
        _logger.debug("window history not updated: %s", e)


def load_peaks(cache_dir: Path) -> dict[str, dict[str, dict[int, float]]]:
    """``{account: {"five_hour": {reset_epoch: peak_pct}, "seven_day": {...}}}``."""
    out: dict[str, dict[str, dict[int, float]]] = {}
    for num, windows_by_key in _read(cache_dir / FILENAME).items():
        if not isinstance(windows_by_key, dict):
            continue
        for key in WINDOW_KEYS:
            windows = windows_by_key.get(key)
            if not isinstance(windows, dict):
                continue
            parsed = {
                int(r): float(p)
                for r, p in windows.items()
                if r.lstrip("-").isdigit()
                and isinstance(p, (int, float))
                and not isinstance(p, bool)
            }
            if parsed:
                out.setdefault(str(num), {})[key] = parsed
    return out
