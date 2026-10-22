"""Token accounting from Claude Code's local session transcripts.

Claude Code appends every API response to ``<config>/projects/<project>/<session>.jsonl``.
Assistant lines carry ``message.usage`` (input / output / cache-write / cache-read
token counts), ``message.model`` and a UTC ``timestamp``. One API response is
written as several lines (one per content block) that all repeat the same usage,
so records are de-duplicated on ``(message.id, requestId)``.

Attribution to accounts:

- Transcripts under a ``cswap run`` session profile
  (``<backup>/sessions/<num>-<slug>/projects``) belong to that slot outright.
- Transcripts under the default profile are attributed with the switch timeline
  parsed from ``claude-swap.log`` (``Switched from account X to Y``): a record
  belongs to the account that was active at its timestamp. Records older than
  the first logged switch go to that switch's *from* account; with no switches
  logged at all they go to ``fallback_account`` (the currently active one).

The log rotates (1 MB x 3 backups), so very old switches can be lost; those
records are then attributed to the earliest known account. This is an estimate
of *local* usage only: other machines and claude.ai chat are not counted.

Pure functions only (no switcher state) so the CLI and tests share them.
"""

from __future__ import annotations

import bisect
import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

GROUP_BY_CHOICES = ("account", "model", "day", "project")

# "2026-10-22 01:24:20,663 - INFO - Switched from account 1 to 2"
_SWITCH_LINE_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3}) - \w+ - "
    r"Switched from account (\d+) to (\d+)"
)
_SESSION_DIR_RE = re.compile(r"^(\d+)-")

UNKNOWN_ACCOUNT = "?"


@dataclass(frozen=True)
class UsageRecord:
    """One de-duplicated API response's token usage."""

    timestamp: datetime  # timezone-aware, UTC
    model: str
    project: str
    input_tokens: int
    output_tokens: int
    cache_creation_tokens: int
    cache_read_tokens: int
    # Slot that owns the transcript (session profile), or None for the
    # default profile, which needs the switch timeline.
    slot: str | None = None


@dataclass
class UsageTotals:
    """Summed token counts for one group."""

    messages: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_creation_tokens
            + self.cache_read_tokens
        )

    def add(self, record: UsageRecord) -> None:
        self.messages += 1
        self.input_tokens += record.input_tokens
        self.output_tokens += record.output_tokens
        self.cache_creation_tokens += record.cache_creation_tokens
        self.cache_read_tokens += record.cache_read_tokens

    def to_json(self) -> dict:
        return {
            "messages": self.messages,
            "inputTokens": self.input_tokens,
            "outputTokens": self.output_tokens,
            "cacheCreationTokens": self.cache_creation_tokens,
            "cacheReadTokens": self.cache_read_tokens,
            "totalTokens": self.total_tokens,
        }


@dataclass
class UsageReport:
    """Grouped totals plus the grand total, sorted by total tokens descending."""

    group_by: str
    rows: list[tuple[str, UsageTotals]] = field(default_factory=list)
    total: UsageTotals = field(default_factory=UsageTotals)


# --------------------------------------------------------------------------- #
# Switch timeline
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SwitchEvent:
    at: datetime  # timezone-aware
    from_account: str
    to_account: str


def parse_switch_timeline(log_text: str) -> list[SwitchEvent]:
    """Switch events from ``claude-swap.log`` text, oldest first.

    Log timestamps are naive local time (``logging`` ``asctime``); they are
    made aware in the local zone so they compare against UTC transcripts.
    Unparseable lines are skipped.
    """
    events: list[SwitchEvent] = []
    for line in log_text.splitlines():
        m = _SWITCH_LINE_RE.match(line)
        if not m:
            continue
        naive = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(
            microsecond=int(m.group(2)) * 1000
        )
        events.append(SwitchEvent(naive.astimezone(), m.group(3), m.group(4)))
    events.sort(key=lambda e: e.at)
    return events


def read_switch_log(backup_dir: Path) -> str:
    """Concatenate the rotated logs oldest-first (``.3`` … ``.1``, then current)."""
    base = backup_dir / "claude-swap.log"
    parts: list[str] = []
    for path in [base.with_name(f"{base.name}.{i}") for i in (3, 2, 1)] + [base]:
        try:
            parts.append(path.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
    return "\n".join(parts)


class AccountTimeline:
    """Answers "which account was active at instant T?" from switch events."""

    def __init__(self, events: list[SwitchEvent], fallback_account: str | None):
        self._events = events
        self._times = [e.at for e in events]
        self._fallback = fallback_account or UNKNOWN_ACCOUNT

    @property
    def has_events(self) -> bool:
        """False when attribution relies on the fallback account alone."""
        return bool(self._events)

    def account_at(self, when: datetime) -> str:
        if not self._events:
            return self._fallback
        idx = bisect.bisect_right(self._times, when)
        if idx == 0:
            return self._events[0].from_account
        return self._events[idx - 1].to_account


# --------------------------------------------------------------------------- #
# Transcript scanning
# --------------------------------------------------------------------------- #


def _parse_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _int(value: object) -> int:
    return value if isinstance(value, int) and value > 0 else 0


def parse_transcript_line(
    line: str, project: str, slot: str | None
) -> tuple[str, UsageRecord] | None:
    """``(dedupe_key, record)`` for an assistant line with usage, else None."""
    if '"usage"' not in line:  # cheap pre-filter; most lines carry no usage
        return None
    try:
        entry = json.loads(line)
    except ValueError:
        return None
    if not isinstance(entry, dict):
        return None
    message = entry.get("message")
    if not isinstance(message, dict):
        return None
    usage = message.get("usage")
    if not isinstance(usage, dict):
        return None
    model = message.get("model")
    if not isinstance(model, str) or model == "<synthetic>":
        return None
    when = _parse_timestamp(entry.get("timestamp"))
    if when is None:
        return None

    message_id = message.get("id")
    request_id = entry.get("requestId")
    if message_id or request_id:
        key = f"{message_id}:{request_id}"
    else:
        key = f"{entry.get('uuid')}"

    record = UsageRecord(
        timestamp=when,
        model=model,
        project=project,
        input_tokens=_int(usage.get("input_tokens")),
        output_tokens=_int(usage.get("output_tokens")),
        cache_creation_tokens=_int(usage.get("cache_creation_input_tokens")),
        cache_read_tokens=_int(usage.get("cache_read_input_tokens")),
        slot=slot,
    )
    return key, record


def _project_name(transcript: Path, projects_dir: Path) -> str:
    """The project folder name (Claude's mangled cwd), or the file's parent."""
    try:
        return transcript.relative_to(projects_dir).parts[0]
    except (ValueError, IndexError):
        return transcript.parent.name


def transcript_roots(
    config_home: Path, backup_dir: Path
) -> list[tuple[Path, str | None]]:
    """``(projects_dir, slot)`` pairs to scan; session profiles come first.

    Session profiles are listed first so that, when ``--share-history`` links a
    profile's ``projects/`` into the default one, the de-duplication keeps the
    profile's exact slot instead of the timeline guess.
    """
    roots: list[tuple[Path, str | None]] = []
    sessions = backup_dir / "sessions"
    try:
        profiles = sorted(p for p in sessions.iterdir() if p.is_dir())
    except OSError:
        profiles = []
    for profile in profiles:
        m = _SESSION_DIR_RE.match(profile.name)
        if m:
            roots.append((profile / "projects", m.group(1)))
    roots.append((config_home / "projects", None))
    return roots


def iter_records(
    roots: Iterable[tuple[Path, str | None]],
    since: datetime | None = None,
    until: datetime | None = None,
) -> Iterator[UsageRecord]:
    """De-duplicated usage records from every transcript under ``roots``.

    Files last modified before ``since`` are skipped without being opened:
    transcripts are append-only, so nothing in them can be newer than their
    mtime. That keeps ``--days 1`` fast on a large history.
    """
    seen: set[str] = set()
    since_ts = since.timestamp() if since else None
    for projects_dir, slot in roots:
        if not projects_dir.is_dir():
            continue
        for transcript in projects_dir.rglob("*.jsonl"):
            if since_ts is not None:
                try:
                    if transcript.stat().st_mtime < since_ts:
                        continue
                except OSError:
                    continue
            project = _project_name(transcript, projects_dir)
            try:
                handle = transcript.open(encoding="utf-8", errors="replace")
            except OSError:
                continue
            with handle:
                for line in handle:
                    parsed = parse_transcript_line(line, project, slot)
                    if parsed is None:
                        continue
                    key, record = parsed
                    if key in seen:
                        continue
                    if since is not None and record.timestamp < since:
                        continue
                    if until is not None and record.timestamp >= until:
                        continue
                    seen.add(key)
                    yield record


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


def aggregate(
    records: Iterable[UsageRecord],
    group_by: str,
    timeline: AccountTimeline,
    account_label: dict[str, str] | None = None,
) -> UsageReport:
    """Sum ``records`` into groups. ``account_label`` maps slot → display name."""
    if group_by not in GROUP_BY_CHOICES:
        raise ValueError(f"group_by must be one of {GROUP_BY_CHOICES}")
    labels = account_label or {}
    groups: dict[str, UsageTotals] = {}
    report = UsageReport(group_by=group_by)
    for record in records:
        if group_by == "account":
            slot = record.slot or timeline.account_at(record.timestamp)
            key = labels.get(slot, slot)
        elif group_by == "model":
            key = record.model
        elif group_by == "day":
            key = record.timestamp.astimezone().date().isoformat()
        else:
            key = record.project
        groups.setdefault(key, UsageTotals()).add(record)
        report.total.add(record)
    if group_by == "day":
        report.rows = sorted(groups.items())
    else:
        report.rows = sorted(groups.items(), key=lambda kv: -kv[1].total_tokens)
    return report


def window_start(days: int | None, since: date | None) -> datetime | None:
    """Local-midnight start of the window as an aware datetime.

    ``days=1`` means "today", ``days=7`` the last seven calendar days including
    today. ``since`` wins when both are given.
    """
    if since is not None:
        start = since
    elif days is not None:
        start = date.fromordinal(date.today().toordinal() - (days - 1))
    else:
        return None
    return datetime(start.year, start.month, start.day).astimezone()


def format_tokens(n: int) -> str:
    """Compact human count: 950, 12.3K, 4.56M, 1.20B."""
    if n < 1000:
        return str(n)
    for div, suffix in ((1_000_000_000, "B"), (1_000_000, "M"), (1_000, "K")):
        if n >= div:
            value = n / div
            return f"{value:.2f}{suffix}" if value < 100 else f"{value:.1f}{suffix}"
    return str(n)  # unreachable
