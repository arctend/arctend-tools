"""The command line, the client and the package hygiene."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
from conftest import ROOT, FakeArctend, make_question

from arctend import __main__ as cli
from arctend.api import (
    DEFAULT_API_URL,
    USER_AGENT,
    VERSION,
    ArctendClient,
    RedirectRefusedError,
)


@pytest.fixture
def keyed_env(fake: FakeArctend, monkeypatch: pytest.MonkeyPatch) -> FakeArctend:
    monkeypatch.setenv("ARCTEND_API_URL", fake.url)
    monkeypatch.setenv("ARCTEND_API_KEY", fake.api_key)
    monkeypatch.chdir(ROOT)  # so examples.stub imports from the current directory
    return fake


def test_check_prints_me_and_the_first_page(
    keyed_env: FakeArctend, capsys: pytest.CaptureFixture[str]
) -> None:
    for n in (1, 2, 3):
        keyed_env.add(make_question("binary", n))
    assert cli.run(["check"]) == 0
    out = capsys.readouterr().out
    assert "Agent: test-bot (Test bot), tier provisional" in out
    assert "Open questions, first page: 2 (more pages)" in out
    assert "Question 1?" in out and "Question 3?" not in out
    assert all(r["method"] == "GET" for r in keyed_env.requests)
    assert [r["headers"]["User-Agent"] for r in keyed_env.requests] == [USER_AGENT] * 2


def test_check_without_a_key_exits_2(
    fake: FakeArctend, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("ARCTEND_API_URL", fake.url)
    monkeypatch.delenv("ARCTEND_API_KEY", raising=False)
    assert cli.run(["check"]) == 2
    assert "ARCTEND_API_KEY" in capsys.readouterr().err
    assert fake.requests == []


def test_run_with_the_shipped_stub_bot(
    keyed_env: FakeArctend, capsys: pytest.CaptureFixture[str]
) -> None:
    keyed_env.add(make_question("binary", 1))
    keyed_env.add(make_question("multiple_choice", 2))
    keyed_env.add(make_question("numeric", 3))
    assert cli.run(["run", "--bot", "examples.stub:UniformBot", "--dry-run"]) == 0
    assert keyed_env.puts() == []
    assert "Done: 3 computed (dry run)." in capsys.readouterr().out
    assert cli.run(["run", "--bot", "examples.stub:UniformBot"]) == 0
    bodies = [p["body"] for p in keyed_env.puts()]
    assert len(bodies) == 3
    assert bodies[0]["probability"] == 0.5
    assert bodies[1]["vector"] == pytest.approx([1 / 3] * 3)
    assert len(bodies[2]["vector"]) == 5 and sum(bodies[2]["vector"]) == pytest.approx(1)
    assert all(b["agent_version"].startswith("UniformBot via ") for b in bodies)
    out = capsys.readouterr().out
    assert "Done: 3 submitted." in out
    # The stub bot does not skip by default; refresh is the bot's decision.
    kwargs = '{"skip_previously_forecasted_questions": true}'
    assert cli.run(["run", "--bot", "examples.stub:UniformBot", "--kwargs", kwargs]) == 0
    assert len(keyed_env.puts()) == 3
    assert "Done: 3 skipped." in capsys.readouterr().out


def test_run_stopped_by_a_rejected_key_exits_2(
    keyed_env: FakeArctend, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("ARCTEND_API_KEY", "atk_wrong.key")
    assert cli.run(["run", "--bot", "examples.stub:UniformBot"]) == 2
    assert "rejected" in capsys.readouterr().err


def test_run_rejects_a_class_that_is_not_a_bot(keyed_env: FakeArctend) -> None:
    with pytest.raises(ValueError, match="ForecastBot"):
        cli.run(["run", "--bot", "arctend.api:ArctendClient"])
    with pytest.raises(ValueError, match="MODULE:CLASS"):
        cli.run(["run", "--bot", "examples.stub"])
    with pytest.raises(ValueError, match="JSON object"):
        cli.run(["run", "--bot", "examples.stub:UniformBot", "--kwargs", "[1]"])


def test_register_help_says_the_name_is_public(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        cli.run(["register", "--help"])
    assert "public name" in capsys.readouterr().out


@pytest.mark.parametrize("command", ["check", "register --name Plumbing", "--help"])
def test_check_and_register_never_import_forecasting_tools(
    command: str, fake: FakeArctend, tmp_path: Path
) -> None:
    fake.add(make_question("binary", 1))
    code = (
        "import sys, runpy\n"
        f"sys.argv = ['arctend'] + {command.split()!r}\n"
        "try:\n"
        "    runpy.run_module('arctend', run_name='__main__')\n"
        "except SystemExit as exit:\n"
        "    assert exit.code in (0, None), exit.code\n"
        "assert 'forecasting_tools' not in sys.modules, 'forecasting_tools was imported'\n"
        "print('clean')\n"
    )
    env = dict(os.environ, ARCTEND_API_URL=fake.url, ARCTEND_API_KEY=fake.api_key)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
        timeout=60,
    )
    assert "clean" in proc.stdout, proc.stderr
    assert list(tmp_path.iterdir()) == []
    if command != "--help":
        assert fake.requests, "the command made no request"


# --- Client -------------------------------------------------------------------------------------


def test_from_env_defaults_to_the_sandbox() -> None:
    client = ArctendClient.from_env({})
    assert client.base_url == DEFAULT_API_URL == "https://sandbox.arctend.com"
    assert not client.has_key
    keyed = ArctendClient.from_env(
        {"ARCTEND_API_URL": "https://api.arctend.com/", "ARCTEND_API_KEY": " atk_x "}
    )
    assert keyed.base_url == "https://api.arctend.com" and keyed.has_key
    with pytest.raises(ValueError):
        ArctendClient("ftp://example.org")


def test_redirects_are_refused(fake: FakeArctend) -> None:
    fake.script[("GET", "/v1/me")] = [(302, {"Location": "https://evil.example/v1/me"}, {})]
    with pytest.raises(RedirectRefusedError) as err:
        ArctendClient(fake.url, fake.api_key).me()
    assert err.value.location_host == "evil.example"


def test_long_retry_after_is_not_waited(fake: FakeArctend) -> None:
    fake.script[("GET", "/v1/me")] = [(429, {"Retry-After": "3600"}, {"code": "rate_limited"})]
    sleeps: list[float] = []
    with pytest.raises(Exception, match="429"):
        ArctendClient(fake.url, fake.api_key, sleep=sleeps.append).me()
    assert sleeps == []


# --- Hygiene ------------------------------------------------------------------------------------


def test_version_matches_pyproject() -> None:
    meta = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert meta["version"] == VERSION
    assert USER_AGENT == f"arctend-python/{VERSION}"
    assert meta["requires-python"] == ">=3.11,<3.14"
    assert meta["dependencies"] == ["forecasting-tools>=0.3.5,<0.4"]


def _shipped_text_files() -> list[Path]:
    files = [ROOT / "README.md", ROOT / "pyproject.toml"]
    for folder in ("arctend", "examples"):
        files += [p for p in (ROOT / folder).rglob("*") if p.suffix in (".py", ".yaml", ".md")]
    return files


BANNED_COPY = re.compile(
    r"\b(bets?|betting|wagers?|odds|stakes?|staking|gambl\w*|win money|cash out|payouts?|"
    r"prizes?|guaranteed|certified|risk-free|most accurate)\b",
    re.IGNORECASE,
)


def test_readme_follows_the_copy_rules() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert not BANNED_COPY.findall(text)
    assert "beat your Metaculus" not in text
    assert "second feed" in text or "second question feed" in text


# Commit trailers and tool credits (CLAUDE.md, git attribution rule), assembled from parts so
# this file does not itself contain them.
FORBIDDEN_MARKERS = ("-".join(("Co", "Authored", "By")), " ".join(("Generated", "with")))


def test_no_attribution_trailers_in_the_package() -> None:
    for path in _shipped_text_files():
        text = path.read_text(encoding="utf-8")
        for marker in FORBIDDEN_MARKERS:
            assert marker.lower() not in text.lower(), path


def test_no_metaculus_token_is_read() -> None:
    for path in (ROOT / "arctend").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert 'METACULUS_TOKEN"' not in text and 'getenv("METACULUS' not in text, path
        assert "MetaculusClient(" not in text, path
