"""Session history: every 5h / 7d window per account, side by side.

One row per window with the tokens this machine's Claude Code spent in it,
the window's peak utilization (recorded by ``window_history`` from every
usage reading) and their ratio, tokens per 1% of the limit — the number that
makes windows and accounts comparable. Windows from before peaks were
recorded are estimated from the transcripts alone (marked ``~``).

The scan runs in a thread worker over a 30-day ``TranscriptIndex`` owned by
the screen, incremental after the first pass.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Static

from claude_swap import token_usage, window_history
from claude_swap.paths import get_claude_config_home
from claude_swap.token_usage import SessionWindow, format_tokens

if TYPE_CHECKING:
    from claude_swap.tui.app import CswapApp

HISTORY_DAYS = 30
REFRESH_S = 60.0
_WINDOW_LABELS = {"five_hour": "5h", "seven_day": "7d"}
COLUMNS = ("Acct", "Start", "End", "Msgs", "Total", "Out", "Peak", "Tok / 1%")


def _stamp(epoch: float) -> str:
    return datetime.fromtimestamp(epoch).strftime("%d/%m %H:%M")


def history_rows(windows: list[SessionWindow], now: float) -> list[tuple[str, ...]]:
    """Table cells for each window (newest first, as given)."""
    rows = []
    for w in windows:
        marker = "" if w.exact else "~"
        end = "now" if w.start <= now < w.end else _stamp(w.end)
        ratio = w.tokens_per_pct
        rows.append((
            w.account,
            f"{marker}{_stamp(w.start)}",
            end,
            str(w.totals.messages),
            format_tokens(w.totals.total_tokens),
            format_tokens(w.totals.output_tokens),
            f"{w.peak_pct:g}%" if w.peak_pct is not None else "—",
            format_tokens(int(ratio)) if ratio is not None else "—",
        ))
    return rows


def account_averages(windows: list[SessionWindow], now: float) -> dict[str, dict]:
    """Per account, over *closed* windows with local usage: count, mean
    tokens, mean output, and mean tokens per 1% where a peak is known."""
    out: dict[str, dict] = {}
    for w in windows:
        if w.end > now or w.totals.messages == 0:
            continue
        acc = out.setdefault(w.account, {"n": 0, "total": 0, "out": 0, "ratios": []})
        acc["n"] += 1
        acc["total"] += w.totals.total_tokens
        acc["out"] += w.totals.output_tokens
        if w.tokens_per_pct is not None:
            acc["ratios"].append(w.tokens_per_pct)
    for acc in out.values():
        n = acc["n"]
        acc["avg_total"] = acc["total"] / n
        acc["avg_out"] = acc["out"] / n
        ratios = acc.pop("ratios")
        acc["avg_ratio"] = sum(ratios) / len(ratios) if ratios else None
    return out


def summary_text(averages: dict[str, dict], labels: dict[str, str]) -> str:
    if not averages:
        return "no closed windows with local usage yet"
    parts = []
    for num in sorted(averages, key=lambda n: (not n.isdigit(), int(n) if n.isdigit() else 0, n)):
        a = averages[num]
        ratio = f" · {format_tokens(int(a['avg_ratio']))}/1%" if a["avg_ratio"] else ""
        parts.append(
            f"{labels.get(num, num)}: avg {format_tokens(int(a['avg_total']))}"
            f" ({format_tokens(int(a['avg_out']))} out){ratio} over {a['n']}"
        )
    return "   ".join(parts)


class HistoryScreen(Screen):
    """Every rate-limit window, per account, with tokens and peak usage."""

    app: "CswapApp"

    BINDINGS = [
        Binding("t", "toggle_window", "5h / 7d"),
        Binding("a", "cycle_account", "Account"),
        Binding("escape,q", "app.pop_screen", "Back"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._window_key = "five_hour"
        self._account: str | None = None  # None = all accounts
        self._index = token_usage.TranscriptIndex(horizon_s=HISTORY_DAYS * 86400)
        self._records: list = []
        self._loading = False

    def compose(self) -> ComposeResult:
        yield Static("", id="list-title")
        yield Static("", id="history-summary")
        yield DataTable(id="history", zebra_stripes=True, cursor_type="row")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#history", DataTable).add_columns(*COLUMNS)
        self._update_title("loading…")
        self._load()
        self.set_interval(REFRESH_S, self._load)

    # -- data ------------------------------------------------------------------

    def _load(self) -> None:
        if self._loading:
            return
        self._loading = True
        self.run_worker(
            self._load_blocking, thread=True, group="history", exit_on_error=False
        )

    def _load_blocking(self) -> None:
        backup_dir = self.app.switcher.backup_dir
        records = self._index.refresh(
            token_usage.transcript_roots(get_claude_config_home(), backup_dir),
            time.time(),
        )
        self.app.call_from_thread(self._loaded, records)

    def _loaded(self, records: list) -> None:
        self._loading = False
        self._records = records
        self._render_table()

    def on_worker_state_changed(self, event) -> None:
        if event.worker.group == "history" and event.worker.error is not None:
            self._loading = False
            self._update_title(f"could not read transcripts: {event.worker.error}")

    def _windows(self, now: float) -> list[SessionWindow]:
        backup_dir = self.app.switcher.backup_dir
        snap = self.app.snapshot
        active = snap.active_number if snap is not None else None
        timeline = token_usage.AccountTimeline(
            token_usage.parse_switch_timeline(token_usage.read_switch_log(backup_dir)),
            str(active) if active is not None else None,
        )
        known: dict[str, dict[int, float | None]] = {}
        for num, by_key in window_history.load_peaks(backup_dir / "cache").items():
            known[num] = dict(by_key.get(self._window_key, {}))
        # The live reading names the current window even before it is recorded.
        for acc in snap.accounts if snap is not None else ():
            window = (acc.usage.last_good or {}).get(self._window_key)
            if isinstance(window, dict):
                reset = window_history.reset_key(window.get("resets_at"))
                pct = window.get("pct")
                if reset is not None:
                    resets = known.setdefault(acc.number, {})
                    if isinstance(pct, (int, float)):
                        resets[reset] = max(pct, resets.get(reset) or 0.0)
                    else:
                        resets.setdefault(reset, None)
        cutoff = now - HISTORY_DAYS * 86400
        windows = token_usage.session_history(
            self._records, timeline, self._window_key, known
        )
        return [w for w in windows if w.end >= cutoff]

    # -- render ----------------------------------------------------------------

    def _labels(self) -> dict[str, str]:
        snap = self.app.snapshot
        return {
            acc.number: f"{acc.number} {acc.alias or acc.email.split('@')[0]}"
            for acc in (snap.accounts if snap is not None else ())
        }

    def _render_table(self) -> None:
        now = time.time()
        windows = self._windows(now)
        shown = [w for w in windows if self._account in (None, w.account)]
        table = self.query_one("#history", DataTable)
        table.clear()
        table.add_rows(history_rows(shown, now))
        self.query_one("#history-summary", Static).update(
            summary_text(account_averages(windows, now), self._labels())
        )
        self._update_title()

    def _update_title(self, note: str = "") -> None:
        label = _WINDOW_LABELS[self._window_key]
        account = f"account {self._account}" if self._account else "all accounts"
        tail = f" · {note}" if note else ""
        self.query_one("#list-title", Static).update(
            f"session history · {label} windows · {account} · last {HISTORY_DAYS} days"
            f" · ~ = estimated{tail}"
        )

    # -- actions ---------------------------------------------------------------

    def action_toggle_window(self) -> None:
        self._window_key = "seven_day" if self._window_key == "five_hour" else "five_hour"
        self._render_table()

    def action_cycle_account(self) -> None:
        snap = self.app.snapshot
        numbers = [acc.number for acc in (snap.accounts if snap is not None else ())]
        order: list[str | None] = [None, *numbers]
        try:
            position = order.index(self._account)
        except ValueError:
            position = 0
        self._account = order[(position + 1) % len(order)]
        self._render_table()
