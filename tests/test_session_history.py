"""Session history: recorded window peaks, window reconstruction, TUI screen."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from claude_swap import window_history
from claude_swap.token_usage import (
    AccountTimeline,
    SessionWindow,
    UsageRecord,
    UsageTotals,
    session_history,
)
from claude_swap.tui.history import account_averages, history_rows, summary_text
from claude_swap.usage_store import FetchRecord, UsageStore
from claude_swap.window_history import load_peaks, note_usage, reset_key

from tests.test_token_usage import _line
from tests.test_tui import FakeSwitcher, make_account, make_app, settle

H = 3600
NOW = 1_999_999_980.0  # on a whole minute: reset keys round to the minute


def _iso(epoch: float, micro: int = 0) -> str:
    dt = datetime.fromtimestamp(epoch, timezone.utc).replace(microsecond=micro)
    return dt.isoformat()


def _usage(pct5: float, reset5: float, pct7: float = 1.0, reset7: float = NOW + 86400):
    return {
        "five_hour": {"pct": pct5, "resets_at": _iso(reset5, 104586)},
        "seven_day": {"pct": pct7, "resets_at": _iso(reset7)},
    }


def _rec(epoch: float, total: int = 10, slot: str | None = None) -> UsageRecord:
    return UsageRecord(
        timestamp=datetime.fromtimestamp(epoch, timezone.utc), model="m", project="p",
        input_tokens=0, output_tokens=total, cache_creation_tokens=0,
        cache_read_tokens=0, slot=slot,
    )


class TestWindowPeaks:
    def test_reset_key_rounds_to_minute(self):
        assert reset_key(_iso(NOW + 20, 104586)) == NOW
        assert reset_key("2026-10-22T07:40:00.104586+00:00") % 60 == 0
        assert reset_key(None) is None
        assert reset_key("garbage") is None

    def test_keeps_running_max_per_window(self, tmp_path):
        note_usage(tmp_path, {"1": _usage(20, NOW)}, NOW - H)
        note_usage(tmp_path, {"1": _usage(55, NOW)}, NOW - 60)
        note_usage(tmp_path, {"1": _usage(40, NOW)}, NOW - 30)  # lower: ignored
        note_usage(tmp_path, {"1": _usage(5, NOW + 5 * H)}, NOW + 60)  # next window
        peaks = load_peaks(tmp_path)
        assert peaks["1"]["five_hour"] == {int(NOW): 55.0, int(NOW + 5 * H): 5.0}
        assert peaks["1"]["seven_day"] == {int(NOW + 86400): 1.0}

    def test_ignores_garbage_and_prunes_old(self, tmp_path):
        note_usage(tmp_path, {"1": None, "2": {"five_hour": {"pct": True,
                                                             "resets_at": _iso(NOW)}}}, NOW)
        assert not (tmp_path / window_history.FILENAME).exists()
        old = NOW - window_history.RETENTION_S - 86400
        note_usage(tmp_path, {"1": _usage(10, old)}, old)
        note_usage(tmp_path, {"1": _usage(10, NOW)}, NOW)
        assert list(load_peaks(tmp_path)["1"]["five_hour"]) == [int(NOW)]

    def test_corrupt_file_is_replaced(self, tmp_path):
        (tmp_path / window_history.FILENAME).write_text("{nope", encoding="utf-8")
        assert load_peaks(tmp_path) == {}
        note_usage(tmp_path, {"1": _usage(10, NOW)}, NOW)
        assert load_peaks(tmp_path)["1"]["five_hour"][int(NOW)] == 10.0

    def test_usage_store_record_feeds_peaks(self, tmp_path):
        store = UsageStore(tmp_path / "cache", clock=lambda: NOW)
        ident = {"1": ("a@x.com", ""), "2": ("b@x.com", "")}
        store.record(
            {"1": FetchRecord(usage=_usage(42, NOW + H)),
             "2": FetchRecord(error="http-429")},
            ident,
        )
        peaks = load_peaks(tmp_path / "cache")
        assert peaks["1"]["five_hour"] == {int(NOW + H): 42.0}
        assert "2" not in peaks
        stored = json.loads(store.path.read_text(encoding="utf-8"))
        assert stored  # the usage row itself is still written


class TestSessionHistory:
    def test_exact_windows_from_known_resets(self):
        known = {"1": {int(NOW): 80.0, int(NOW - 5 * H): 30.0}}
        records = [_rec(NOW - H), _rec(NOW - 2 * H), _rec(NOW - 6 * H, total=5)]
        windows = session_history(records, AccountTimeline([], "1"), "five_hour", known)
        assert [(w.start, w.exact) for w in windows] == [
            (NOW - 5 * H, True), (NOW - 10 * H, True)
        ]
        assert windows[0].totals.total_tokens == 20
        assert windows[0].peak_pct == 80.0
        assert windows[0].tokens_per_pct == pytest.approx(0.25)
        assert windows[1].totals.total_tokens == 5

    def test_estimates_unknown_windows(self):
        records = [_rec(NOW - 20 * H), _rec(NOW - 18 * H), _rec(NOW - 14 * H)]
        windows = session_history(records, AccountTimeline([], "1"), "five_hour", {})
        assert [(w.start, w.end, w.totals.messages) for w in windows] == [
            (NOW - 14 * H, NOW - 9 * H, 1),
            (NOW - 20 * H, NOW - 15 * H, 2),
        ]
        assert not any(w.exact for w in windows)
        assert windows[0].tokens_per_pct is None

    def test_estimate_stops_at_next_known_window(self):
        known = {"1": {int(NOW): None}}  # known window starts NOW - 5h
        records = [_rec(NOW - 7 * H), _rec(NOW - 4 * H)]
        windows = session_history(records, AccountTimeline([], "1"), "five_hour", known)
        estimate = next(w for w in windows if not w.exact)
        assert estimate.end == NOW - 5 * H
        assert estimate.totals.messages == 1

    def test_accounts_kept_apart(self):
        records = [_rec(NOW - H, slot="1"), _rec(NOW - H, slot="2"), _rec(NOW - H / 2, slot="2")]
        windows = session_history(records, AccountTimeline([], None), "five_hour", {})
        by_account = {w.account: w.totals.messages for w in windows}
        assert by_account == {"1": 1, "2": 2}

    def test_known_window_without_local_usage_is_listed(self):
        windows = session_history([], AccountTimeline([], "1"), "seven_day",
                                  {"2": {int(NOW): 64.0}})
        assert len(windows) == 1
        assert windows[0].totals.messages == 0
        assert windows[0].peak_pct == 64.0


class TestHistoryFormatting:
    def _w(self, account, start, end, total, peak=None, exact=True):
        t = UsageTotals(messages=1, cache_read_tokens=total, output_tokens=0)
        return SessionWindow(account, start, end, exact, t, peak)

    def test_rows(self):
        rows = history_rows(
            [self._w("1", NOW - H, NOW + 4 * H, 2_000_000, 40.0),
             self._w("2", NOW - 9 * H, NOW - 4 * H, 500, exact=False)],
            NOW,
        )
        assert rows[0][0] == "1"
        assert rows[0][2] == "now"
        assert rows[0][6:] == ("40%", "50.00K")
        assert rows[1][1].startswith("~")
        assert rows[1][6:] == ("—", "—")

    def test_averages_skip_open_and_empty_windows(self):
        windows = [
            self._w("1", NOW - H, NOW + H, 999),  # still open
            self._w("1", NOW - 10 * H, NOW - 5 * H, 100, 10.0),
            self._w("1", NOW - 20 * H, NOW - 15 * H, 300, None),
            SessionWindow("2", NOW - 9 * H, NOW - 4 * H, True, UsageTotals(), 50.0),
        ]
        avgs = account_averages(windows, NOW)
        assert set(avgs) == {"1"}
        assert avgs["1"]["n"] == 2
        assert avgs["1"]["avg_total"] == 200
        assert avgs["1"]["avg_ratio"] == 10.0
        text = summary_text(avgs, {"1": "1 work"})
        assert text.startswith("1 work: avg 200")
        assert "10/1%" in text
        assert summary_text({}, {}) == "no closed windows with local usage yet"


@pytest.mark.asyncio
class TestHistoryScreen:
    async def test_opens_from_dashboard_and_toggles(self, temp_home: Path):
        now = time.time()
        transcript = temp_home / ".claude" / "projects" / "p" / "s.jsonl"
        transcript.parent.mkdir(parents=True)
        transcript.write_text(
            _line(datetime.fromtimestamp(now - 3 * 86400, timezone.utc).isoformat(),
                  msg_id="old") + "\n"
            + _line(datetime.fromtimestamp(now - 600, timezone.utc).isoformat(),
                    msg_id="new") + "\n",
            encoding="utf-8",
        )
        backup = temp_home / "backup"
        (backup / "cache").mkdir(parents=True)
        fake = FakeSwitcher([make_account(1, active=True), make_account(2)], backup)
        app = make_app(fake)
        async with app.run_test(size=(120, 40)) as pilot:
            await settle(pilot)
            await pilot.press("h")
            await pilot.pause()
            await settle(pilot)
            from textual.widgets import DataTable, Static

            from claude_swap.tui.history import HistoryScreen

            assert isinstance(app.screen, HistoryScreen)
            table = app.screen.query_one("#history", DataTable)
            # 5h: the current window (live reset) plus an estimated old one;
            # account 2's live window is listed with no local usage.
            assert table.row_count == 3
            title = str(app.screen.query_one("#list-title", Static).render())
            assert "5h windows" in title

            await pilot.press("t")
            await pilot.pause()
            assert "7d windows" in str(app.screen.query_one("#list-title", Static).render())

            await pilot.press("a")  # account 1 only
            await pilot.pause()
            accounts = {table.get_row_at(i)[0] for i in range(table.row_count)}
            assert accounts == {"1"}

            await pilot.press("escape")
            await pilot.pause()
            assert not isinstance(app.screen, HistoryScreen)

    async def test_menu_entry(self, temp_home: Path):
        from tests.test_tui import menu_select

        backup = temp_home / "backup"
        backup.mkdir()
        app = make_app(FakeSwitcher([make_account(1, active=True)], backup))
        async with app.run_test(size=(120, 40)) as pilot:
            await settle(pilot)
            await menu_select(pilot, "history")
            from claude_swap.tui.history import HistoryScreen

            assert isinstance(app.screen, HistoryScreen)
