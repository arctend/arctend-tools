"""The read commands against the local fake server: readable output and ``--json``."""

from __future__ import annotations

import json
import sys

import pytest
from conftest import AGENT_ID, FakeArctend, consensus_point, make_question

from arctend import __main__ as cli


@pytest.fixture
def keyed(fake: FakeArctend, monkeypatch: pytest.MonkeyPatch) -> FakeArctend:
    monkeypatch.setenv("ARCTEND_API_URL", fake.url)
    monkeypatch.setenv("ARCTEND_API_KEY", fake.api_key)
    return fake


def run(argv: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, str, str]:
    code = cli.run(argv)
    out = capsys.readouterr()
    return code, out.out, out.err


def run_main(
    argv: list[str], capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> tuple[int, str, str]:
    """Through ``main()``, so ApiError handling and the exit code are covered."""
    monkeypatch.setattr(sys, "argv", ["arctend", *argv])
    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    out = capsys.readouterr()
    return int(exit_info.value.code or 0), out.out, out.err


def test_questions_lists_one_line_each_and_honours_limit(
    keyed: FakeArctend, capsys: pytest.CaptureFixture[str]
) -> None:
    for n in (1, 2, 3):
        keyed.add(make_question("binary", n))
    code, out, _ = run(["questions"], capsys)
    lines = out.strip().splitlines()
    assert code == 0 and len(lines) == 3
    assert "binary" in lines[0] and "UTC" in lines[0] and "Question 1?" in lines[0]
    assert lines[0].startswith(str(make_question("binary", 1)["id"]))
    _, out, _ = run(["questions", "--limit", "2"], capsys)
    assert len(out.strip().splitlines()) == 2
    _, out, _ = run(["questions", "--limit", "2", "--topic", "macro", "--json"], capsys)
    assert [q["title"] for q in json.loads(out)] == ["Question 1?", "Question 2?"]
    assert keyed.requests[-1]["query"]["topic"] == ["macro"]


def test_questions_sends_no_authorization_header_without_a_key(
    fake: FakeArctend,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("ARCTEND_API_URL", fake.url)
    monkeypatch.delenv("ARCTEND_API_KEY", raising=False)
    code, _, err = run_main(["questions"], capsys, monkeypatch)
    assert code == 1 and "401" in err  # the fake wants a key; the point is that none was sent
    assert "Authorization" not in fake.requests[-1]["headers"]


def test_question_shows_times_and_resolution_rules(
    keyed: FakeArctend, capsys: pytest.CaptureFixture[str]
) -> None:
    q = keyed.add(make_question("multiple_choice", 4))
    code, out, _ = run(["question", q["id"]], capsys)
    assert code == 0
    assert "title: Question 4?" in out and "format: multiple_choice" in out
    assert "opens:" in out and "closes:" in out and "resolves:" in out
    assert "option 0: Red" in out
    assert "YES if the published rate is above 4." in out
    assert "Central bank" in out
    _, out, _ = run(["question", q["id"], "--json"], capsys)
    assert json.loads(out)["id"] == q["id"]


def test_consensus_binary_multiple_choice_and_numeric(
    keyed: FakeArctend, capsys: pytest.CaptureFixture[str]
) -> None:
    b = keyed.add(make_question("binary", 1))
    m = keyed.add(make_question("multiple_choice", 2))
    n = keyed.add(make_question("numeric", 3))
    keyed.consensus[b["id"]] = consensus_point([0.7, 0.3])
    keyed.consensus[m["id"]] = consensus_point([0.5, 0.3, 0.2])
    keyed.consensus[n["id"]] = consensus_point([0.1, 0.2, 0.3, 0.3, 0.1], thin_crowd=False)

    _, out, _ = run(["consensus", b["id"]], capsys)
    assert "probability: 0.7000" in out and "n_forecasters: 7" in out
    assert "n_attested: 3" in out and "thin_crowd: yes" in out
    assert "computed_at: 2026-11-07T12:00:00.000Z" in out

    _, out, _ = run(["consensus", m["id"]], capsys)
    assert "Red: 0.5000" in out and "Blue: 0.3000" in out and "Green: 0.2000" in out

    _, out, _ = run(["consensus", n["id"]], capsys)
    assert "(-infinity, 0): 0.1000" in out and "[60, +infinity): 0.1000" in out
    assert "thin_crowd: no" in out

    _, out, _ = run(["consensus", b["id"], "--json"], capsys)
    assert json.loads(out)["median"]["value"] == [0.7, 0.3]


def test_consensus_before_any_exists_and_history(
    keyed: FakeArctend, capsys: pytest.CaptureFixture[str]
) -> None:
    q = keyed.add(make_question("binary", 1))
    _, out, _ = run(["consensus", q["id"]], capsys)
    assert out.startswith("No consensus yet")
    keyed.history[q["id"]] = [
        consensus_point([0.7, 0.3], computed_at="2026-11-07T12:00:00.000Z"),
        consensus_point([0.6, 0.4], computed_at="2026-11-06T12:00:00.000Z"),
    ]
    code, out, _ = run(["consensus", q["id"], "--history"], capsys)
    lines = out.strip().splitlines()
    assert code == 0 and len(lines) == 2
    assert lines[0].startswith("2026-11-07T12:00:00.000Z") and "0.7000" in lines[0]
    _, out, _ = run(["consensus", q["id"], "--history", "--json"], capsys)
    assert len(json.loads(out)["items"]) == 2
    assert keyed.requests[-1]["path"].endswith("/consensus/history")


def test_me_prints_key_value_lines(keyed: FakeArctend, capsys: pytest.CaptureFixture[str]) -> None:
    _, out, _ = run(["me"], capsys)
    assert "tier: provisional" in out and "slug: test-bot" in out
    assert "forecast_writes_per_day: 200" in out
    _, out, _ = run(["me", "--json"], capsys)
    assert json.loads(out)["agent"]["id"] == AGENT_ID


def test_rating_with_and_without_an_id(
    keyed: FakeArctend, capsys: pytest.CaptureFixture[str]
) -> None:
    keyed.ratings[AGENT_ID] = {
        "agent_id": AGENT_ID,
        "slug": "test-bot",
        "view": "public",
        "overall": {"scope": "overall", "status": "provisional"},
        "topics": [],
        "badges": [],
    }
    code, out, _ = run(["rating"], capsys)  # own agent, found through /v1/me
    assert code == 0 and "slug: test-bot" in out and "status: provisional" in out
    assert keyed.paths()[-2:] == ["/v1/me", f"/v1/agents/{AGENT_ID}/rating"]
    _, out, _ = run(["rating", AGENT_ID, "--json"], capsys)
    assert json.loads(out)["slug"] == "test-bot"


def test_rating_not_published_is_a_message_not_a_failure(
    keyed: FakeArctend, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, err = run(["rating", "00000000-0000-0000-0000-00000000ffff"], capsys)
    assert code == 0 and "No published rating for this agent" in out and err == ""


def test_api_errors_print_status_and_code_and_exit_1(
    keyed: FakeArctend, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    keyed.script[("GET", "/v1/me")] = [
        (429, {"Retry-After": "3600"}, {"code": "rate_limited", "detail": "Slow down"})
    ]
    code, _, err = run_main(["me"], capsys, monkeypatch)
    assert code == 1
    assert "429 rate_limited: Slow down" in err and "retry after 3600" in err
