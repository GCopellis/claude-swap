"""Tokens per rate-limit window: token_usage helpers and the TUI watch view."""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from claude_swap.token_usage import (
    WINDOW_SPANS,
    AccountTimeline,
    SwitchEvent,
    TranscriptIndex,
    UsageRecord,
    UsageTotals,
    window_starts,
    window_totals,
)
from claude_swap.tui.theme import Palette
from claude_swap.tui.widgets import tokens_line_text

from tests.test_token_usage import _line
from tests.test_tui import FakeSwitcher, make_account, make_app, settle

NOW = 2_000_000_000.0  # 2033-05-18


def _iso_ts(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat().replace("+00:00", "Z")


def _rec(epoch: float, out: int = 1, slot: str | None = None) -> UsageRecord:
    return UsageRecord(
        timestamp=datetime.fromtimestamp(epoch, timezone.utc),
        model="m", project="p", input_tokens=0, output_tokens=out,
        cache_creation_tokens=0, cache_read_tokens=0, slot=slot,
    )


class TestWindowStarts:
    def test_from_future_resets(self):
        last_good = {
            "five_hour": {"pct": 10, "resets_at": _iso_ts(NOW + 7200)},
            "seven_day": {"pct": 10, "resets_at": _iso_ts(NOW + 86400)},
        }
        starts = window_starts(last_good, NOW)
        assert starts["five_hour"] == NOW + 7200 - WINDOW_SPANS["five_hour"]
        assert starts["seven_day"] == NOW + 86400 - WINDOW_SPANS["seven_day"]

    def test_past_reset_counts_since_the_reset(self):
        last_good = {"five_hour": {"pct": 0, "resets_at": _iso_ts(NOW - 600)}}
        assert window_starts(last_good, NOW)["five_hour"] == NOW - 600

    @pytest.mark.parametrize(
        "last_good",
        [None, {}, {"five_hour": {"pct": 0}}, {"five_hour": {"resets_at": "junk"}}],
    )
    def test_unknown_reset_uses_rolling_span(self, last_good):
        starts = window_starts(last_good, NOW)
        assert starts == {k: NOW - span for k, span in WINDOW_SPANS.items()}


class TestWindowTotals:
    def test_attributes_and_filters_per_window(self):
        timeline = AccountTimeline([SwitchEvent(
            datetime.fromtimestamp(NOW - 3600, timezone.utc), "1", "2"
        )], None)
        records = [
            _rec(NOW - 7200, out=1),     # account 1 (before switch), 7d only
            _rec(NOW - 1800, out=10),    # account 2, both windows
            _rec(NOW - 60, out=100, slot="1"),  # session profile → account 1
            _rec(NOW - 60, out=1000, slot="9"),  # unknown account: ignored
        ]
        starts = {
            "1": {"five_hour": NOW - 3600, "seven_day": NOW - 86400},
            "2": {"five_hour": NOW - 3600, "seven_day": NOW - 86400},
        }
        out = window_totals(records, timeline, starts)
        assert out["1"]["five_hour"].output_tokens == 100
        assert out["1"]["seven_day"].output_tokens == 101
        assert out["2"]["five_hour"].output_tokens == 10
        assert out["2"]["seven_day"].messages == 1
        assert "9" not in out


class TestTranscriptIndex:
    def test_reads_only_appended_complete_lines(self, tmp_path):
        projects = tmp_path / "projects"
        path = projects / "p" / "s.jsonl"
        path.parent.mkdir(parents=True)
        now = time.time()
        first = _line(_iso_ts(now - 60), msg_id="a")
        path.write_text(first + "\n", encoding="utf-8")
        index = TranscriptIndex()
        assert len(index.refresh([(projects, None)], now)) == 1

        partial = _line(_iso_ts(now - 30), msg_id="b")
        with path.open("a", encoding="utf-8") as handle:
            handle.write(first + "\n")  # repeated content block: de-duped
            handle.write(partial[:40])  # line still being written
        assert len(index.refresh([(projects, None)], now)) == 1

        with path.open("a", encoding="utf-8") as handle:
            handle.write(partial[40:] + "\n")
        records = index.refresh([(projects, None)], now)
        assert len(records) == 2

    def test_prunes_beyond_horizon_and_skips_old_files(self, tmp_path):
        projects = tmp_path / "projects"
        now = time.time()
        old = projects / "p" / "old.jsonl"
        old.parent.mkdir(parents=True)
        old.write_text(_line(_iso_ts(now - 30 * 86400)) + "\n", encoding="utf-8")
        os.utime(old, (now - 30 * 86400, now - 30 * 86400))
        recent = projects / "p" / "new.jsonl"
        recent.write_text(
            _line(_iso_ts(now - 3600), msg_id="x") + "\n"
            + _line(_iso_ts(now - 9 * 86400), msg_id="y") + "\n",
            encoding="utf-8",
        )
        index = TranscriptIndex()
        records = index.refresh([(projects, None)], now)
        assert [r.timestamp.timestamp() for r in records] == [pytest.approx(now - 3600)]
        # A later refresh moves the cutoff and drops what fell out.
        assert index.refresh([(projects, None)], now + 8 * 86400) == []

    def test_rewritten_file_is_reread(self, tmp_path):
        projects = tmp_path / "projects"
        path = projects / "p" / "s.jsonl"
        path.parent.mkdir(parents=True)
        now = time.time()
        path.write_text(
            (_line(_iso_ts(now - 60), msg_id="a") + "\n") * 3, encoding="utf-8"
        )
        index = TranscriptIndex()
        index.refresh([(projects, None)], now)
        path.write_text(_line(_iso_ts(now - 10), msg_id="b") + "\n", encoding="utf-8")
        assert len(index.refresh([(projects, None)], now)) == 2


def test_tokens_line_text():
    five = UsageTotals(messages=1, output_tokens=85_200, cache_read_tokens=12_214_800)
    seven = UsageTotals(messages=2, output_tokens=2_710_000)
    text = tokens_line_text(
        {"five_hour": five, "seven_day": seven}, palette=Palette.DARK
    ).plain
    assert text.startswith("tokens  ")
    assert "12.30M (85.20K out)" in text
    assert "7d 2.71M (2.71M out)" in text
    assert tokens_line_text({}, palette=Palette.DARK).plain == "tokens"


@pytest.mark.asyncio
class TestWatchTokens:
    async def test_watch_cards_show_tokens_per_window(self, temp_home: Path):
        now = time.time()
        transcript = temp_home / ".claude" / "projects" / "proj" / "s.jsonl"
        transcript.parent.mkdir(parents=True)
        transcript.write_text(
            _line(_iso_ts(now - 3600), msg_id="a", inp=0, out=500, cw=0, cr=0) + "\n"
            + _line(_iso_ts(now - 2 * 86400), msg_id="b", inp=0, out=7000, cw=0, cr=0)
            + "\n",
            encoding="utf-8",
        )
        backup = temp_home / "backup"
        backup.mkdir()
        fake = FakeSwitcher([make_account(1, active=True), make_account(2)], backup)
        app = make_app(fake)
        async with app.run_test(size=(110, 40)) as pilot:
            await settle(pilot)
            await pilot.press("w")
            await pilot.pause()
            await settle(pilot)
            from claude_swap.tui.widgets import AccountCard

            windows = app.token_windows
            # No switch log: default-profile usage belongs to the active account.
            assert windows["1"]["five_hour"].output_tokens == 500
            assert windows["1"]["seven_day"].output_tokens == 7500
            assert windows["2"]["seven_day"].messages == 0
            cards = list(app.screen.query(AccountCard))
            rendered = cards[0].render().plain
            assert "tokens" in rendered
            assert "500 out" in rendered
            assert "7.50K out" in rendered

    async def test_dashboard_alone_does_not_count(self, temp_home: Path):
        backup = temp_home / "backup"
        backup.mkdir()
        app = make_app(FakeSwitcher([make_account(1, active=True)], backup))
        async with app.run_test(size=(110, 40)) as pilot:
            await settle(pilot)
            assert app._token_index is None
            assert app.token_windows == {}
