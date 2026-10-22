"""Tests for resuming limit-stopped conversations (resume.py + `cswap auto --autostart`)."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from claude_swap import cli, resume
from claude_swap.autoswitch import ResumeEvent, TickOutcome
from claude_swap.resume import (
    DEFAULT_RESUME_PROMPT,
    PausedSession,
    ResumeResult,
    build_resume_argv,
    find_claude_binary,
    find_paused_sessions,
    is_limit_pause,
    last_message,
    resume_env,
    resume_session,
)
from claude_swap.settings import load_settings, merged_with_cli

from tests.test_autoswitch import EngineHarness, _iso_at, _usage


def _iso(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


def _user(ts: str, uuid: str = "u1", sid: str = "sess-1", cwd: str = "/work") -> str:
    return json.dumps({
        "type": "user", "uuid": uuid, "timestamp": ts, "sessionId": sid, "cwd": cwd,
        "isSidechain": False, "message": {"role": "user", "content": "do it"},
    })


def _pause(ts: str, uuid: str = "p1", sid: str = "sess-1", cwd: str = "/work") -> str:
    return json.dumps({
        "type": "assistant", "uuid": uuid, "timestamp": ts, "sessionId": sid,
        "cwd": cwd, "isSidechain": False, "apiError": "usage_limit_reached",
        "error": "rate_limit", "isApiErrorMessage": True,
        "message": {
            "model": "<synthetic>", "role": "assistant",
            "content": [{"type": "text", "text": "You've hit your session limit"}],
        },
    })


def _write(path: Path, *lines: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _session(tmp_path: Path, **kw) -> PausedSession:
    base = dict(
        session_id="sess-1",
        cwd=str(tmp_path),
        paused_at=datetime(2026, 10, 22, tzinfo=timezone.utc),
        pause_uuid="p1",
        transcript=tmp_path / "s.jsonl",
        config_dir=tmp_path / ".claude",
        slot=None,
    )
    base.update(kw)
    return PausedSession(**base)


class TestDetection:
    def test_is_limit_pause(self):
        assert is_limit_pause(json.loads(_pause("2026-10-22T10:00:00Z")))
        assert is_limit_pause({"type": "assistant", "isApiErrorMessage": True,
                               "error": "rate_limit"})
        assert not is_limit_pause({"type": "assistant", "isApiErrorMessage": True,
                                   "error": "server_error"})
        assert not is_limit_pause(json.loads(_user("2026-10-22T10:00:00Z")))

    def test_last_message_skips_metadata_and_sidechains(self):
        ts = "2026-10-22T10:00:00Z"
        sidechain = json.loads(_user(ts, uuid="side"))
        sidechain["isSidechain"] = True
        lines = [
            _pause(ts),
            json.dumps(sidechain),
            json.dumps({"type": "last-prompt", "lastPrompt": "x"}),
            json.dumps({"type": "cost-state", "totalCostUSD": 1}),
            "garbage {",
        ]
        assert last_message(lines)["uuid"] == "p1"

    def test_find_paused_sessions(self, tmp_path):
        now = datetime.now(timezone.utc)
        projects = tmp_path / "projects"
        _write(projects / "a" / "s1.jsonl",
               _user(_iso(now - timedelta(hours=1))),
               _pause(_iso(now - timedelta(minutes=30)), sid="s1"))
        # The user answered after the pause: not paused any more.
        _write(projects / "a" / "s2.jsonl",
               _pause(_iso(now - timedelta(minutes=30)), uuid="p2", sid="s2"),
               _user(_iso(now - timedelta(minutes=10)), uuid="u2", sid="s2"))
        # Paused, but long before the window.
        _write(projects / "b" / "s3.jsonl",
               _pause(_iso(now - timedelta(days=3)), uuid="p3", sid="s3"))
        # Subagent transcripts one level deeper are ignored.
        _write(projects / "a" / "s1" / "subagents" / "agent.jsonl",
               _pause(_iso(now - timedelta(minutes=5)), uuid="p4", sid="s4"))

        found = find_paused_sessions([(projects, None)], now - timedelta(hours=6))
        assert [s.session_id for s in found] == ["s1"]
        assert found[0].pause_uuid == "p1"
        assert found[0].cwd == "/work"
        assert found[0].config_dir == tmp_path

    def test_profile_root_wins_and_keeps_slot(self, tmp_path):
        ts = _iso(datetime.now(timezone.utc))
        profile = tmp_path / "sessions" / "2-b" / "projects"
        default = tmp_path / "default" / "projects"
        _write(profile / "p" / "s.jsonl", _pause(ts))
        _write(default / "p" / "s.jsonl", _pause(ts))
        found = find_paused_sessions(
            [(profile, "2"), (default, None)],
            datetime.now(timezone.utc) - timedelta(hours=1),
        )
        assert len(found) == 1
        assert found[0].slot == "2"

    def test_tail_read_drops_partial_first_line(self, tmp_path, monkeypatch):
        ts = _iso(datetime.now(timezone.utc))
        pause = _pause(ts)
        # The tail starts mid-way through the user lines and still holds the
        # whole pause line.
        monkeypatch.setattr(resume, "_TAIL_BYTES", len(pause) + 50)
        path = _write(tmp_path / "p" / "s.jsonl", _user(ts), _user(ts), pause)
        assert resume.last_message(resume._tail_lines(path))["uuid"] == "p1"


class TestLaunch:
    def test_argv(self, tmp_path):
        s = _session(tmp_path)
        assert build_resume_argv("claude", s, "go", None) == [
            "claude", "--bg", "--resume", "sess-1", "go"
        ]
        assert build_resume_argv("claude", s, "go", "acceptEdits") == [
            "claude", "--bg", "--resume", "sess-1", "--permission-mode",
            "acceptEdits", "go",
        ]

    def test_env_sets_config_dir_only_for_profiles(self, tmp_path):
        default = tmp_path / ".claude"
        assert resume_env(_session(tmp_path), default).get(
            "CLAUDE_CONFIG_DIR"
        ) == os.environ.get("CLAUDE_CONFIG_DIR")
        profile = tmp_path / "sessions" / "2-b"
        env = resume_env(_session(tmp_path, config_dir=profile, slot="2"), default)
        assert env["CLAUDE_CONFIG_DIR"] == str(profile)

    def test_resume_session_runs_in_cwd(self, tmp_path):
        calls = []

        def runner(argv, **kw):
            calls.append((argv, kw))
            return subprocess.CompletedProcess(argv, 0, stdout="started a1b2\n", stderr="")

        result = resume_session(
            _session(tmp_path), "claude", default_config_home=tmp_path / ".claude",
            runner=runner,
        )
        assert result.ok
        assert result.detail == "started a1b2"
        argv, kw = calls[0]
        assert argv[-1] == DEFAULT_RESUME_PROMPT
        assert kw["cwd"] == str(tmp_path)
        assert kw["stdin"] is subprocess.DEVNULL

    def test_resume_session_reports_failure(self, tmp_path):
        def runner(argv, **kw):
            return subprocess.CompletedProcess(
                argv, 1, stdout="", stderr="Workspace not trusted."
            )

        result = resume_session(
            _session(tmp_path), "claude", default_config_home=tmp_path, runner=runner
        )
        assert not result.ok
        assert "not trusted" in result.detail

    def test_resume_session_missing_cwd(self, tmp_path):
        def runner(argv, **kw):  # pragma: no cover - must not run
            raise AssertionError("launched")

        result = resume_session(
            _session(tmp_path, cwd=str(tmp_path / "gone")), "claude",
            default_config_home=tmp_path, runner=runner,
        )
        assert not result.ok
        assert "gone" in result.detail

    def test_resume_session_launch_error(self, tmp_path):
        def runner(argv, **kw):
            raise FileNotFoundError("claude")

        result = resume_session(
            _session(tmp_path), "claude", default_config_home=tmp_path, runner=runner
        )
        assert not result.ok
        assert "FileNotFoundError" in result.detail

    def test_find_claude_binary(self, tmp_path):
        exe = tmp_path / "claude-custom"
        exe.write_text("")
        assert find_claude_binary(str(exe)) == str(exe)
        with patch("shutil.which", return_value="/usr/bin/claude"):
            assert find_claude_binary() == "/usr/bin/claude"

    def test_find_claude_binary_falls_back_to_vscode_extension(self, temp_home):
        name = "claude.exe" if os.name == "nt" else "claude"
        old = (temp_home / ".vscode" / "extensions" / "anthropic.claude-code-2.1.1"
               / "resources" / "native-binary" / name)
        new = (temp_home / ".vscode" / "extensions" / "anthropic.claude-code-2.1.9"
               / "resources" / "native-binary" / name)
        for p in (old, new):
            p.parent.mkdir(parents=True)
            p.write_text("")
        os.utime(old, (1, 1))
        with patch("shutil.which", return_value=None):
            assert find_claude_binary() == str(new)
        with patch("shutil.which", return_value=None), patch("glob.glob", return_value=[]):
            assert find_claude_binary() is None


# --------------------------------------------------------------------------- #
# Engine integration
# --------------------------------------------------------------------------- #


class ResumeHarness(EngineHarness):
    def __init__(self, temp_home: Path, **settings):
        super().__init__(temp_home, **settings)
        self.launched: list[PausedSession] = []
        self.engine = self._make_engine(resume_launcher=self._launch)
        for num, email in ((1, "a@example.com"), (2, "b@example.com")):
            self.seed(num, email)
        self.make_live("a@example.com", 1)

    def _launch(self, session: PausedSession) -> ResumeResult:
        self.launched.append(session)
        return ResumeResult(session, True, "started")

    def pause_conversation(self, uuid: str = "p1", sid: str = "sess-1") -> None:
        _write(
            self.temp_home / ".claude" / "projects" / "proj" / f"{sid}.jsonl",
            _user(_iso_at(self.clock.now - 120), sid=sid),
            _pause(_iso_at(self.clock.now - 60), uuid=uuid, sid=sid),
        )

    def resumed(self) -> list[ResumeEvent]:
        return [e for e in self.events if isinstance(e, ResumeEvent)]


RESET = "2099-01-01T00:00:00Z"
EXHAUSTED = {"1": _usage(100, RESET), "2": _usage(100, RESET)}


class TestEngineResume:
    def test_resumes_after_all_exhausted_then_switch(self, temp_home):
        h = ResumeHarness(temp_home, resume_paused=True, cooldown_seconds=0.0)
        h.pause_conversation()

        assert h.tick_with_usage(EXHAUSTED) is TickOutcome.BLOCKED
        assert "all-exhausted" in h.kinds()
        assert h.state()["exhaustedSince"] == h.clock.now
        assert h.launched == []

        h.clock.advance(3600)
        outcome = h.tick_with_usage({"1": _usage(100, RESET), "2": _usage(3)})
        assert outcome is TickOutcome.SWITCHED
        assert h.active_number() == 2
        assert [s.session_id for s in h.launched] == ["sess-1"]
        [event] = h.resumed()
        assert event.ok and event.to_json()["event"] == "session-resumed"
        state = h.state()
        assert "exhaustedSince" not in state
        assert "p1" in state["resumedPauses"]

    def test_resumes_when_active_account_recovers_without_switch(self, temp_home):
        h = ResumeHarness(temp_home, resume_paused=True)
        h.pause_conversation()
        h.tick_with_usage(EXHAUSTED)
        h.clock.advance(3600)
        outcome = h.tick_with_usage({"1": _usage(5), "2": _usage(100, RESET)})
        assert outcome is TickOutcome.NO_ACTION
        assert h.active_number() == 1
        assert len(h.launched) == 1

    def test_no_resume_while_still_exhausted(self, temp_home):
        h = ResumeHarness(temp_home, resume_paused=True)
        h.pause_conversation()
        h.tick_with_usage(EXHAUSTED)
        h.clock.advance(600)
        h.tick_with_usage(EXHAUSTED)
        assert h.launched == []
        assert h.state()["exhaustedSince"] == h.clock.now - 600

    def test_each_pause_resumed_once(self, temp_home):
        h = ResumeHarness(temp_home, resume_paused=True, cooldown_seconds=0.0)
        h.pause_conversation()
        h.tick_with_usage({"1": _usage(100, RESET), "2": _usage(3)})
        assert len(h.launched) == 1
        # Transcript still ends in the pause (background resume not written
        # yet); a later switch must not launch it again.
        h.clock.advance(600)
        h.tick_with_usage({"1": _usage(3), "2": _usage(100, RESET)})
        assert len(h.launched) == 1

    def test_failed_launch_is_not_retried(self, temp_home):
        h = ResumeHarness(temp_home, resume_paused=True, cooldown_seconds=0.0)
        h._launch = lambda s: ResumeResult(s, False, "Workspace not trusted.")
        h.engine = h._make_engine(resume_launcher=h._launch)
        h.pause_conversation()
        h.tick_with_usage({"1": _usage(100, RESET), "2": _usage(3)})
        [event] = h.resumed()
        assert not event.ok
        assert "not trusted" in event.human()
        assert "p1" in h.state()["resumedPauses"]

    def test_resume_max_caps_a_pass(self, temp_home):
        h = ResumeHarness(temp_home, resume_paused=True, resume_max=2,
                          cooldown_seconds=0.0)
        for i in range(4):
            h.pause_conversation(uuid=f"p{i}", sid=f"s{i}")
        h.tick_with_usage({"1": _usage(100, RESET), "2": _usage(3)})
        assert len(h.launched) == 2

    def test_disabled_by_default(self, temp_home):
        h = ResumeHarness(temp_home, cooldown_seconds=0.0)
        h.pause_conversation()
        h.tick_with_usage(EXHAUSTED)
        h.tick_with_usage({"1": _usage(100, RESET), "2": _usage(3)})
        assert h.launched == []
        assert "exhaustedSince" not in h.state()

    def test_dry_run_reports_without_launching(self, temp_home):
        h = ResumeHarness(temp_home, resume_paused=True, cooldown_seconds=0.0)
        h.engine = h._make_engine(resume_launcher=h._launch, dry_run=True)
        h.pause_conversation()
        h.tick_with_usage({"1": _usage(100, RESET), "2": _usage(3)})
        [event] = h.resumed()
        assert event.dry_run
        assert "would resume" in event.human()
        assert h.launched == []
        assert h.state() == {}

    def test_launcher_crash_does_not_break_tick(self, temp_home):
        h = ResumeHarness(temp_home, resume_paused=True, cooldown_seconds=0.0)

        def boom(session):
            raise RuntimeError("boom")

        h.engine = h._make_engine(resume_launcher=boom)
        h.pause_conversation()
        assert h.tick_with_usage(
            {"1": _usage(100, RESET), "2": _usage(3)}
        ) is TickOutcome.SWITCHED
        assert any(e.kind == "error" and "boom" in e.message for e in h.events)


class TestAutostartFlag:
    @pytest.mark.parametrize("flag, expected", [("y", True), ("n", False)])
    def test_flag_overrides_setting(self, temp_home, flag, expected):
        captured = {}

        class FakeEngine:
            def __init__(self, switcher, settings, on_event, **kw):
                captured["settings"] = settings

            def tick(self):
                return TickOutcome.NO_ACTION

        with patch("claude_swap.autoswitch.AutoSwitchEngine", FakeEngine), \
                pytest.raises(SystemExit):
            cli._auto_command(["--once", "--autostart", flag])
        assert captured["settings"].resume_paused is expected

    def test_flag_absent_keeps_setting(self, temp_home):
        class Args:
            resume_paused = None

        base = load_settings(temp_home)
        assert merged_with_cli(base, Args()).resume_paused is False

    def test_rejects_other_values(self, temp_home):
        with pytest.raises(SystemExit):
            cli._auto_command(["--once", "--autostart", "maybe"])
