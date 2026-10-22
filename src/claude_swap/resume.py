"""Find Claude Code conversations stopped by a usage limit, and resume them.

When a request is rejected for quota, Claude Code ends the turn with a
synthetic assistant message carrying ``"apiError": "usage_limit_reached"``
(``"error": "rate_limit"``, ``"isApiErrorMessage": true``) and the
conversation sits there until the user types again. A transcript whose last
main-chain message is that error is a *paused* conversation.

Resuming uses Claude Code's own background mode::

    claude --bg --resume <session-id> [--permission-mode M] "<prompt>"

run in the conversation's original ``cwd``. It continues the session under the
same ID (``claude agents`` lists it, ``claude attach`` opens it). If the
conversation is still open in a terminal or IDE panel, Claude Code starts a
copy instead and says so — cswap never writes into a live session itself.

The auto-switch engine decides *when* (see ``AutoSwitchEngine``); this module
only finds candidates and launches the resume.
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

DEFAULT_RESUME_PROMPT = (
    "The previous turn was interrupted because the account hit its usage "
    "limit. Quota is available again: continue the task exactly from where "
    "you stopped, without redoing finished steps."
)

# Only the tail of a transcript is needed to find its last message; a few
# hundred KB covers even a large tool result as the final entry.
_TAIL_BYTES = 512 * 1024
_MESSAGE_TYPES = ("user", "assistant")
_LAUNCH_TIMEOUT_S = 60


@dataclass(frozen=True)
class PausedSession:
    session_id: str
    cwd: str
    paused_at: datetime  # timezone-aware, UTC
    pause_uuid: str  # uuid of the limit-error entry; the de-dupe key
    transcript: Path
    config_dir: Path  # CLAUDE_CONFIG_DIR the conversation lives in
    slot: str | None  # session-profile slot, None for the default profile


@dataclass(frozen=True)
class ResumeResult:
    session: PausedSession
    ok: bool
    detail: str


def is_limit_pause(entry: dict) -> bool:
    """Whether a transcript entry is Claude Code's usage-limit stop message."""
    if entry.get("type") != "assistant":
        return False
    if entry.get("apiError") == "usage_limit_reached":
        return True
    return bool(entry.get("isApiErrorMessage")) and entry.get("error") == "rate_limit"


def _tail_lines(path: Path) -> list[str]:
    try:
        with path.open("rb") as handle:
            size = handle.seek(0, os.SEEK_END)
            start = max(0, size - _TAIL_BYTES)
            handle.seek(start)
            data = handle.read()
    except OSError:
        return []
    lines = data.decode("utf-8", errors="replace").splitlines()
    if start > 0 and lines:
        lines = lines[1:]  # first line is cut mid-record
    return lines


def last_message(lines: Iterable[str]) -> dict | None:
    """The last main-chain user/assistant entry (sidechains and metadata skipped)."""
    found = None
    for line in lines:
        if '"type"' not in line:
            continue
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if (
            isinstance(entry, dict)
            and entry.get("type") in _MESSAGE_TYPES
            and not entry.get("isSidechain")
        ):
            found = entry
    return found


def _parse_ts(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def find_paused_sessions(
    roots: Iterable[tuple[Path, str | None]], since: datetime
) -> list[PausedSession]:
    """Conversations under ``roots`` whose last message is a limit pause at or
    after ``since``, oldest pause first.

    ``roots`` are ``(projects_dir, slot)`` pairs as built by
    ``token_usage.transcript_roots``. Only top-level session files are read
    (subagent transcripts live one level deeper), and files untouched since
    ``since`` are skipped by mtime.
    """
    since_ts = since.timestamp()
    found: dict[str, PausedSession] = {}
    for projects_dir, slot in roots:
        if not projects_dir.is_dir():
            continue
        for transcript in projects_dir.glob("*/*.jsonl"):
            try:
                if transcript.stat().st_mtime < since_ts:
                    continue
            except OSError:
                continue
            entry = last_message(_tail_lines(transcript))
            if entry is None or not is_limit_pause(entry):
                continue
            paused_at = _parse_ts(entry.get("timestamp"))
            session_id = entry.get("sessionId") or transcript.stem
            cwd = entry.get("cwd")
            if paused_at is None or paused_at < since or not isinstance(cwd, str):
                continue
            # A --share-history profile links its projects/ into the default
            # one; the first root (the profile) keeps the exact slot.
            found.setdefault(
                session_id,
                PausedSession(
                    session_id=session_id,
                    cwd=cwd,
                    paused_at=paused_at,
                    pause_uuid=str(entry.get("uuid") or session_id),
                    transcript=transcript,
                    config_dir=projects_dir.parent,
                    slot=slot,
                ),
            )
    return sorted(found.values(), key=lambda s: s.paused_at)


def find_claude_binary(override: str | None = None) -> str | None:
    """The ``claude`` executable: explicit override, PATH, then the newest
    native binary bundled with the VS Code / Cursor extension."""
    if override:
        return override if Path(override).exists() else shutil.which(override)
    on_path = shutil.which("claude")
    if on_path:
        return on_path
    exe = "claude.exe" if sys.platform == "win32" else "claude"
    candidates: list[str] = []
    for editor in (".vscode", ".cursor", ".windsurf", ".vscode-insiders"):
        pattern = str(
            Path.home() / editor / "extensions" / "anthropic.claude-code-*"
            / "resources" / "native-binary" / exe
        )
        candidates.extend(glob.glob(pattern))
    if not candidates:
        return None
    return max(candidates, key=lambda p: os.path.getmtime(p))


def build_resume_argv(
    claude_bin: str,
    session: PausedSession,
    prompt: str,
    permission_mode: str | None,
) -> list[str]:
    argv = [claude_bin, "--bg", "--resume", session.session_id]
    if permission_mode:
        argv += ["--permission-mode", permission_mode]
    argv.append(prompt)
    return argv


def resume_env(session: PausedSession, default_config_home: Path) -> dict[str, str]:
    """Environment for the resume: a session profile's own CLAUDE_CONFIG_DIR,
    otherwise the caller's environment unchanged."""
    env = dict(os.environ)
    if session.config_dir.resolve() != default_config_home.resolve():
        env["CLAUDE_CONFIG_DIR"] = str(session.config_dir)
    return env


Runner = Callable[..., subprocess.CompletedProcess]


def resume_session(
    session: PausedSession,
    claude_bin: str,
    *,
    prompt: str = DEFAULT_RESUME_PROMPT,
    permission_mode: str | None = None,
    default_config_home: Path,
    runner: Runner = subprocess.run,
) -> ResumeResult:
    """Launch the background resume. ``--bg`` returns as soon as the session
    is dispatched, so this waits only for that handshake."""
    if not Path(session.cwd).is_dir():
        return ResumeResult(session, False, f"working directory is gone: {session.cwd}")
    kwargs: dict = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        proc = runner(
            build_resume_argv(claude_bin, session, prompt, permission_mode),
            cwd=session.cwd,
            env=resume_env(session, default_config_home),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_LAUNCH_TIMEOUT_S,
            stdin=subprocess.DEVNULL,
            **kwargs,
        )
    except (OSError, subprocess.SubprocessError) as e:
        return ResumeResult(session, False, f"{type(e).__name__}: {e}")
    output = " ".join((proc.stdout or "").split() + (proc.stderr or "").split())
    return ResumeResult(session, proc.returncode == 0, output[:300])
