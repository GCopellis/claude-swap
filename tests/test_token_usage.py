"""Tests for token accounting from Claude Code transcripts (`cswap tokens`)."""

from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from claude_swap import cli, token_usage
from claude_swap.switcher import ClaudeAccountSwitcher
from claude_swap.token_usage import (
    AccountTimeline,
    SwitchEvent,
    UsageRecord,
    aggregate,
    format_tokens,
    iter_records,
    parse_switch_timeline,
    parse_transcript_line,
    read_switch_log,
    transcript_roots,
    window_start,
)


def _line(
    ts: str,
    *,
    msg_id: str = "msg_1",
    req_id: str = "req_1",
    model: str = "claude-opus-5-5",
    inp: int = 10,
    out: int = 20,
    cw: int = 30,
    cr: int = 40,
) -> str:
    return json.dumps(
        {
            "type": "assistant",
            "timestamp": ts,
            "requestId": req_id,
            "message": {
                "id": msg_id,
                "model": model,
                "role": "assistant",
                "content": [],
                "usage": {
                    "input_tokens": inp,
                    "output_tokens": out,
                    "cache_creation_input_tokens": cw,
                    "cache_read_input_tokens": cr,
                },
            },
        }
    )


def _utc(*args) -> datetime:
    return datetime(*args, tzinfo=timezone.utc)


def _record(ts: datetime, **kw) -> UsageRecord:
    base = dict(
        timestamp=ts,
        model="m",
        project="p",
        input_tokens=1,
        output_tokens=2,
        cache_creation_tokens=3,
        cache_read_tokens=4,
    )
    base.update(kw)
    return UsageRecord(**base)


class TestParseTranscriptLine:
    def test_extracts_usage_and_key(self):
        parsed = parse_transcript_line(_line("2026-10-22T10:00:00.000Z"), "proj", None)
        assert parsed is not None
        key, rec = parsed
        assert key == "msg_1:req_1"
        assert rec.timestamp == _utc(2026, 10, 22, 10)
        assert (rec.input_tokens, rec.output_tokens) == (10, 20)
        assert (rec.cache_creation_tokens, rec.cache_read_tokens) == (30, 40)
        assert rec.model == "claude-opus-5-5"
        assert rec.project == "proj"

    @pytest.mark.parametrize(
        "line",
        [
            "",
            "not json with \"usage\"",
            json.dumps({"type": "user", "message": {"role": "user", "content": "x"}}),
            json.dumps({"timestamp": "2026-10-22T10:00:00Z", "message": {"usage": {}}}),
            _line("2026-10-22T10:00:00Z", model="<synthetic>"),
            _line("garbage"),
        ],
    )
    def test_ignores_non_usage_lines(self, line):
        assert parse_transcript_line(line, "p", None) is None

    def test_negative_or_missing_counts_become_zero(self):
        line = json.loads(_line("2026-10-22T10:00:00Z"))
        line["message"]["usage"] = {"input_tokens": -5, "output_tokens": "7"}
        _, rec = parse_transcript_line(json.dumps(line), "p", None)
        assert rec.input_tokens == 0
        assert rec.output_tokens == 0
        assert rec.cache_read_tokens == 0


class TestSwitchTimeline:
    LOG = (
        "2026-10-22 01:24:20,663 - INFO - Switched from account 1 to 2\n"
        "2026-10-22 01:30:00,000 - INFO - Updated config file\n"
        "2026-10-22 01:32:20,028 - INFO - Switched from account 2 to 1\n"
    )

    def test_parses_switch_lines_in_local_time(self):
        events = parse_switch_timeline(self.LOG)
        assert [(e.from_account, e.to_account) for e in events] == [("1", "2"), ("2", "1")]
        expected = datetime(2026, 10, 22, 1, 24, 20, 663000).astimezone()
        assert events[0].at == expected

    def test_account_at(self):
        t0 = _utc(2026, 1, 1, 10)
        events = [
            SwitchEvent(t0, "1", "2"),
            SwitchEvent(t0 + timedelta(hours=1), "2", "3"),
        ]
        timeline = AccountTimeline(events, fallback_account="9")
        assert timeline.account_at(t0 - timedelta(seconds=1)) == "1"
        assert timeline.account_at(t0) == "2"
        assert timeline.account_at(t0 + timedelta(minutes=30)) == "2"
        assert timeline.account_at(t0 + timedelta(hours=2)) == "3"
        assert timeline.has_events

    def test_fallback_without_events(self):
        assert AccountTimeline([], "4").account_at(_utc(2026, 1, 1)) == "4"
        assert AccountTimeline([], None).account_at(_utc(2026, 1, 1)) == "?"
        assert not AccountTimeline([], "4").has_events

    def test_read_switch_log_includes_rotations_oldest_first(self, tmp_path):
        (tmp_path / "claude-swap.log.2").write_text("old\n", encoding="utf-8")
        (tmp_path / "claude-swap.log.1").write_text("mid\n", encoding="utf-8")
        (tmp_path / "claude-swap.log").write_text("new\n", encoding="utf-8")
        text = read_switch_log(tmp_path)
        assert text.index("old") < text.index("mid") < text.index("new")

    def test_read_switch_log_missing(self, tmp_path):
        assert read_switch_log(tmp_path) == ""


class TestIterRecords:
    def _write(self, path: Path, *lines: str) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def test_dedupes_repeated_content_blocks(self, tmp_path):
        projects = tmp_path / "projects"
        ts = "2026-10-22T10:00:00Z"
        self._write(
            projects / "proj-a" / "s.jsonl",
            _line(ts),
            _line(ts),  # same message, second content block
            _line(ts, msg_id="msg_2", req_id="req_2"),
        )
        recs = list(iter_records([(projects, None)]))
        assert len(recs) == 2
        assert {r.project for r in recs} == {"proj-a"}

    def test_session_profile_wins_on_duplicate(self, tmp_path):
        ts = "2026-10-22T10:00:00Z"
        profile = tmp_path / "sessions" / "3-me_x.com" / "projects"
        default = tmp_path / "default" / "projects"
        self._write(profile / "p" / "s.jsonl", _line(ts))
        self._write(default / "p" / "s.jsonl", _line(ts))
        roots = transcript_roots(tmp_path / "default", tmp_path)
        assert roots[0] == (profile, "3")
        recs = list(iter_records(roots))
        assert len(recs) == 1
        assert recs[0].slot == "3"

    def test_window_filters_records_and_skips_old_files(self, tmp_path):
        projects = tmp_path / "projects"
        old = self._write(projects / "p" / "old.jsonl", _line("2026-01-01T00:00:00Z"))
        os.utime(old, (0, 0))
        self._write(
            projects / "p" / "new.jsonl",
            _line("2026-10-20T00:00:00Z", msg_id="a"),
            _line("2026-10-22T00:00:00Z", msg_id="b"),
        )
        since = _utc(2026, 10, 21)
        with patch.object(Path, "open", wraps=Path.open, autospec=True) as opened:
            recs = list(iter_records([(projects, None)], since=since))
        assert [r.timestamp for r in recs] == [_utc(2026, 10, 22)]
        assert all(call.args[0].name != "old.jsonl" for call in opened.call_args_list)

    def test_missing_root_is_ignored(self, tmp_path):
        assert list(iter_records([(tmp_path / "nope", None)])) == []


class TestAggregate:
    def test_by_account_uses_slot_then_timeline(self):
        t0 = _utc(2026, 10, 22, 12)
        timeline = AccountTimeline([SwitchEvent(t0, "1", "2")], None)
        records = [
            _record(t0 - timedelta(hours=1)),  # before switch → 1
            _record(t0 + timedelta(hours=1)),  # after switch → 2
            _record(t0 + timedelta(hours=1)),
            _record(t0 - timedelta(hours=1), slot="3"),  # profile → 3
        ]
        report = aggregate(records, "account", timeline, {"2": "2: work"})
        assert [k for k, _ in report.rows][0] == "2: work"
        totals = dict(report.rows)
        assert totals["2: work"].messages == 2
        assert totals["1"].messages == 1
        assert totals["3"].messages == 1
        assert report.total.messages == 4
        assert report.total.total_tokens == 4 * 10

    def test_by_model_and_project(self):
        timeline = AccountTimeline([], "1")
        ts = _utc(2026, 10, 22)
        records = [_record(ts, model="a", project="x"), _record(ts, model="b", project="x")]
        assert {k for k, _ in aggregate(records, "model", timeline).rows} == {"a", "b"}
        assert [k for k, _ in aggregate(records, "project", timeline).rows] == ["x"]

    def test_by_day_sorted_chronologically(self):
        timeline = AccountTimeline([], "1")
        records = [_record(_utc(2026, 10, 22, 12)), _record(_utc(2026, 10, 20, 12))]
        keys = [k for k, _ in aggregate(records, "day", timeline).rows]
        assert keys == sorted(keys)
        assert len(keys) == 2

    def test_rejects_unknown_group(self):
        with pytest.raises(ValueError):
            aggregate([], "nope", AccountTimeline([], None))


class TestHelpers:
    @pytest.mark.parametrize(
        "n, expected",
        [(0, "0"), (999, "999"), (1_500, "1.50K"), (123_456, "123.5K"), (4_560_000, "4.56M"),
         (2_000_000_000, "2.00B")],
    )
    def test_format_tokens(self, n, expected):
        assert format_tokens(n) == expected

    def test_window_start(self):
        assert window_start(None, None) is None
        today = date.today()
        assert window_start(1, None).date() == today
        assert window_start(7, None).date() == today - timedelta(days=6)
        assert window_start(7, date(2026, 1, 2)).date() == date(2026, 1, 2)
        assert window_start(1, None).tzinfo is not None


class TestTokensCommand:
    def _seed(self, home: Path) -> ClaudeAccountSwitcher:
        switcher = ClaudeAccountSwitcher()
        switcher._setup_directories()
        switcher._init_sequence_file()
        data = switcher._get_sequence_data()
        data["accounts"]["1"] = {"email": "a@x.com", "uuid": "u1", "added": "2024-01-01T00:00:00Z"}
        data["accounts"]["2"] = {
            "email": "b@x.com", "uuid": "u2", "alias": "work", "added": "2024-01-01T00:00:00Z"
        }
        data["sequence"] = [1, 2]
        data["activeAccountNumber"] = 1
        switcher._write_json(switcher.sequence_file, data)

        now = datetime.now(timezone.utc)
        switch_at = (now - timedelta(minutes=30)).astimezone()
        (switcher.backup_dir / "claude-swap.log").write_text(
            switch_at.strftime("%Y-%m-%d %H:%M:%S,000")
            + " - INFO - Switched from account 1 to 2\n",
            encoding="utf-8",
        )
        transcript = home / ".claude" / "projects" / "proj" / "s.jsonl"
        transcript.parent.mkdir(parents=True)
        iso = lambda d: d.isoformat().replace("+00:00", "Z")  # noqa: E731
        transcript.write_text(
            _line(iso(now - timedelta(hours=1)), msg_id="a", inp=100)
            + "\n"
            + _line(iso(now - timedelta(minutes=5)), msg_id="b", inp=1000)
            + "\n",
            encoding="utf-8",
        )
        return switcher

    def test_json_by_account(self, temp_home, capsys):
        self._seed(temp_home)
        cli._tokens_command(["--json"])
        payload = json.loads(capsys.readouterr().out)
        rows = {r["key"]: r for r in payload["rows"]}
        assert rows["1: a@x.com"]["inputTokens"] == 100
        assert rows["2: work"]["inputTokens"] == 1000
        assert payload["total"]["messages"] == 2
        assert payload["groupBy"] == "account"

    def test_table_output(self, temp_home, capsys):
        self._seed(temp_home)
        cli._tokens_command(["--by", "model", "--days", "1"])
        out = capsys.readouterr().out
        assert "Token usage by model" in out
        assert "claude-opus-5-5" in out
        assert "Total" in out

    def test_empty_window(self, temp_home, capsys):
        self._seed(temp_home)
        cli._tokens_command(["--since", "2999-01-01"])
        assert "No usage recorded" in capsys.readouterr().out

    def test_days_must_be_positive(self, temp_home):
        with pytest.raises(SystemExit):
            cli._tokens_command(["--days", "0"])

    def test_main_dispatches_tokens(self, temp_home, capsys):
        self._seed(temp_home)
        with patch("sys.argv", ["cswap", "tokens", "--json", "--all"]):
            cli.main()
        assert json.loads(capsys.readouterr().out)["since"] is None


def test_group_by_choices_cover_aggregate():
    for group in token_usage.GROUP_BY_CHOICES:
        aggregate([], group, AccountTimeline([], None))
