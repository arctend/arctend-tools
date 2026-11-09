"""``arctend forecast``, ``arctend results``, the console command, network errors and console
encoding, against the local fake server."""

from __future__ import annotations

import importlib.metadata
import json
import os
import ssl
import subprocess
import sys
import urllib.error
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest
from conftest import FakeArctend, make_question

from arctend import __main__ as cli
from arctend import api
from arctend.api import (
    PRODUCTION_API_URL,
    SANDBOX_API_URL,
    ArctendClient,
    idempotency_key,
    manual_idempotency_key,
)


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
    """Through ``main()``, so argparse errors, exception handling and exit codes are covered."""
    monkeypatch.setattr(sys, "argv", ["arctend", *argv])
    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    out = capsys.readouterr()
    return int(exit_info.value.code or 0), out.out, out.err


# --- forecast -----------------------------------------------------------------------------------


def test_forecast_probability_sends_one_put_and_prints_the_receipt(
    keyed: FakeArctend, capsys: pytest.CaptureFixture[str]
) -> None:
    q = keyed.add(make_question("binary", 1))
    code, out, err = run(["forecast", q["id"], "--probability", "0.62"], capsys)
    assert code == 0 and err == ""
    puts = keyed.puts()
    assert len(puts) == 1
    assert puts[0]["body"] == {"probability": 0.62}
    key = puts[0]["headers"]["Idempotency-Key"]
    assert key == manual_idempotency_key(q["id"], {"probability": 0.62})
    assert key.startswith(f"arctend-cli:{q['id']}:")
    assert f"question_id: {q['id']}" in out and "forecast_id: " in out
    assert "replayed: no" in out and "Already submitted" not in out

    code, _, _ = run(
        ["forecast", q["id"], "--probability", "0.4", "--rationale", "Base rate."], capsys
    )
    assert code == 0
    assert keyed.puts()[-1]["body"] == {"probability": 0.4, "rationale": "Base rate."}


def test_forecast_vector_sends_the_vector(
    keyed: FakeArctend, capsys: pytest.CaptureFixture[str]
) -> None:
    q = keyed.add(make_question("multiple_choice", 2))
    code, _, _ = run(["forecast", q["id"], "--vector", "0.2,0.5,0.3"], capsys)
    assert code == 0
    assert keyed.puts()[0]["body"] == {"vector": [0.2, 0.5, 0.3]}


@pytest.mark.parametrize(
    "argv",
    [
        ["forecast", "q1"],
        ["forecast", "q1", "--probability", "0.5", "--vector", "0.5,0.5"],
    ],
)
def test_forecast_needs_exactly_one_of_probability_and_vector(
    argv: list[str],
    keyed: FakeArctend,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    code, _, err = run_main(argv, capsys, monkeypatch)
    assert code == 2 and "usage: arctend forecast" in err
    assert keyed.requests == []


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("--probability", "1.2", "the probability must be a number from 0 to 1"),
        ("--probability", "-0.1", "the probability must be a number from 0 to 1"),
        ("--probability", "nan", "the probability must be a number from 0 to 1"),
        ("--probability", "abc", "the probability must be a number from 0 to 1, not 'abc'"),
        ("--vector", "0.5,0.6", "the vector must sum to 1"),
        ("--vector", "1", "a vector needs at least two entries"),
        ("--vector", "0.5,x", "each vector entry must be a number from 0 to 1, not 'x'"),
        ("--vector", "1.5,-0.5", "each vector entry must be a number from 0 to 1"),
        ("--vector", "inf,0", "each vector entry must be a number from 0 to 1"),
    ],
)
def test_forecast_rejects_bad_values_before_sending(
    option: str,
    value: str,
    message: str,
    keyed: FakeArctend,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    code, out, err = run_main(["forecast", "q1", option, value], capsys, monkeypatch)
    assert code == 1
    assert err.startswith(f"error: {message}") and len(err.strip().splitlines()) == 1
    assert out.startswith("API: ")
    assert keyed.requests == []


def test_forecast_dry_run_prints_the_body_and_sends_nothing(
    keyed: FakeArctend, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, _ = run(
        ["forecast", "q1", "--vector", "0.25,0.75", "--rationale", "Why.", "--dry-run"], capsys
    )
    assert code == 0
    assert keyed.requests == []
    assert "Dry run, nothing sent" in out
    body = json.loads(out[out.index("{") :])
    assert body == {"rationale": "Why.", "vector": [0.25, 0.75]}
    assert manual_idempotency_key("q1", body) in out


def test_forecast_dry_run_needs_no_key(
    fake: FakeArctend, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("ARCTEND_API_URL", fake.url)
    monkeypatch.delenv("ARCTEND_API_KEY", raising=False)
    code, _, _ = run(["forecast", "q1", "--probability", "0.5", "--dry-run"], capsys)
    assert code == 0
    code, _, err = run(["forecast", "q1", "--probability", "0.5"], capsys)
    assert code == 2 and "ARCTEND_API_KEY" in err
    assert fake.requests == []


def test_forecast_same_values_replay_and_new_values_are_a_new_forecast(
    keyed: FakeArctend, capsys: pytest.CaptureFixture[str]
) -> None:
    q = keyed.add(make_question("binary", 1))
    argv = ["forecast", q["id"], "--probability", "0.55"]
    _, first, _ = run(argv, capsys)
    code, second, _ = run(argv, capsys)
    assert code == 0
    assert second.strip().splitlines()[-1] == cli.ALREADY_SUBMITTED
    assert "replayed: yes" in second
    first_id = next(ln for ln in first.splitlines() if ln.startswith("forecast_id:"))
    assert first_id in second  # the original receipt

    code, third, _ = run(["forecast", q["id"], "--probability", "0.6"], capsys)
    assert code == 0 and cli.ALREADY_SUBMITTED not in third
    assert first_id not in third
    keys = [p["headers"]["Idempotency-Key"] for p in keyed.puts()]
    assert keys[0] == keys[1] != keys[2]


def test_forecast_json_prints_the_raw_receipt(
    keyed: FakeArctend, capsys: pytest.CaptureFixture[str]
) -> None:
    q = keyed.add(make_question("binary", 1))
    argv = ["forecast", q["id"], "--probability", "0.3", "--json"]
    _, out, err = run(argv, capsys)
    # stdout is JSON and nothing else, so it can be piped into a parser.
    assert json.loads(out)["probability"] == 0.3
    assert err.splitlines()[0] == f"API: {keyed.url}"
    # A replay with --json: still only the receipt on stdout.
    _, out, _ = run(argv, capsys)
    assert json.loads(out)["replayed"] is True

    _, out, err = run([*argv, "--dry-run"], capsys)
    assert json.loads(out) == {"probability": 0.3}
    assert "Dry run, nothing sent" in err and "Idempotency-Key: arctend-cli:" in err


def test_forecast_first_line_names_the_api(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["forecast", "q1", "--probability", "0.5", "--dry-run"]
    monkeypatch.delenv("ARCTEND_API_URL", raising=False)
    _, out, _ = run(argv, capsys)
    assert out.splitlines()[0] == f"API: {SANDBOX_API_URL} (practice sandbox)"
    monkeypatch.setenv("ARCTEND_API_URL", PRODUCTION_API_URL + "/")
    _, out, _ = run(argv, capsys)
    assert out.splitlines()[0] == f"API: {PRODUCTION_API_URL}"


def test_forecast_shows_the_api_error_when_the_shape_does_not_fit(
    keyed: FakeArctend, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    q = keyed.add(make_question("multiple_choice", 2))
    path = f"/v1/questions/{q['id']}/forecast"
    keyed.script[("PUT", path)] = [
        (400, {}, {"code": "invalid_forecast", "detail": "vector needs 3 entries"})
    ]
    code, _, err = run_main(["forecast", q["id"], "--probability", "0.5"], capsys, monkeypatch)
    assert code == 1 and "400 invalid_forecast: vector needs 3 entries" in err


# --- the client and the key ---------------------------------------------------------------------


def test_manual_idempotency_key() -> None:
    day = datetime(2026, 11, 9, 23, 30, tzinfo=UTC)
    qid = "0f2c1a9e-1111-4222-8333-944455556666"
    key = manual_idempotency_key(qid, {"probability": 0.62}, now=day)
    assert key == manual_idempotency_key(qid, {"probability": 0.62}, now=day)
    assert key.startswith(f"arctend-cli:{qid}:2026-11-09:")
    digest = key.rsplit(":", 1)[1]
    assert len(digest) == 16 and int(digest, 16) >= 0
    assert key != manual_idempotency_key(qid, {"probability": 0.6}, now=day)
    assert key != manual_idempotency_key(
        qid, {"probability": 0.62}, now=datetime(2026, 11, 10, 0, 30, tzinfo=UTC)
    )
    # Key order in the body does not matter; the content does.
    a = manual_idempotency_key(qid, {"vector": [0.5, 0.5], "rationale": "x"}, now=day)
    b = manual_idempotency_key(qid, {"rationale": "x", "vector": [0.5, 0.5]}, now=day)
    assert a == b
    assert len(key) <= 128 and " " not in key
    assert all(33 <= ord(c) <= 126 for c in key)


def test_the_adapter_key_is_unchanged() -> None:
    day = datetime(2026, 11, 9, 23, 30, tzinfo=UTC)
    assert idempotency_key("q-1", day) == "arctend-ft:q-1:2026-11-09"


def test_client_submit_forecast_and_practice_results(keyed: FakeArctend) -> None:
    q = keyed.add(make_question("numeric", 3))
    client = ArctendClient(keyed.url, keyed.api_key)
    vector = [0.1, 0.2, 0.3, 0.3, 0.1]
    receipt = client.submit_forecast(q["id"], vector=vector, rationale="Spread.")
    assert receipt["replayed"] is False
    put = keyed.puts()[0]
    assert put["body"] == {"vector": vector, "rationale": "Spread."}
    assert put["headers"]["Idempotency-Key"] == manual_idempotency_key(q["id"], put["body"])
    assert client.submit_forecast(q["id"], vector=vector, rationale="Spread.")["replayed"] is True
    with pytest.raises(ValueError, match="exactly one"):
        client.submit_forecast(q["id"])
    with pytest.raises(ValueError, match="exactly one"):
        client.submit_forecast(q["id"], probability=0.5, vector=[0.5, 0.5])
    with pytest.raises(ValueError, match="must be a number"):
        client.submit_forecast(q["id"], probability=True)
    with pytest.raises(ValueError, match="from 0 to 1"):
        client.submit_forecast(q["id"], vector=[0.5, 0.5, float("nan")])
    assert len(keyed.puts()) == 2

    keyed.practice_results = [practice_result(n) for n in range(5)]
    assert [r["question_title"] for r in client.practice_results()] == [
        f"Practice {n}?" for n in range(5)
    ]
    page = client.practice_results_page()
    assert len(page["items"]) == 2 and page["next_cursor"] == "2"
    assert client.practice_results_page("4")["items"][0]["question_title"] == "Practice 4?"


def test_the_package_exports_nothing_new() -> None:
    import arctend

    assert arctend.__all__ == ["VERSION", "ApiError", "ArctendClient", "RedirectRefusedError"]


# --- results ------------------------------------------------------------------------------------


def practice_result(n: int, **overrides: object) -> dict[str, object]:
    r: dict[str, object] = {
        "label": "practice",
        "practice": True,
        "question_id": f"00000000-0000-0000-0000-0000000001{n:02d}",
        "question_title": f"Practice {n}?",
        "format": "binary",
        "forecast_received_at": f"2026-11-09T10:{n:02d}:00.000Z",
        "vector": [0.6, 0.4],
        "outcome_index": 0,
        "true_probabilities": [0.5, 0.5],
        "log_score": -0.5108256237659907,
        "computed_at": f"2026-11-09T11:{n:02d}:00.000Z",
    }
    r.update(overrides)
    return r


def test_results_readable_json_and_limit(
    keyed: FakeArctend, capsys: pytest.CaptureFixture[str]
) -> None:
    keyed.practice_results = [
        practice_result(0),
        practice_result(1, format="multiple_choice", outcome_index=2, log_score=None),
        practice_result(2),
    ]
    code, out, _ = run(["results"], capsys)
    lines = out.strip().splitlines()
    assert code == 0 and len(lines) == 3
    assert lines[0].startswith("2026-11-09 10:00 UTC  binary")
    assert "log_score -0.5108" in lines[0] and "outcome 0" in lines[0]
    assert lines[0].endswith("Practice 0?")
    assert "multiple_choice" in lines[1] and "log_score -" in lines[1] and "outcome 2" in lines[1]
    assert [r["path"] for r in keyed.requests] == ["/v1/practice/results"] * 2  # two pages

    keyed.requests.clear()
    _, out, _ = run(["results", "--limit", "2"], capsys)
    assert len(out.strip().splitlines()) == 2
    assert len(keyed.requests) == 1  # the limit stops before the next page

    _, out, _ = run(["results", "--json", "--limit", "1"], capsys)
    assert json.loads(out) == [practice_result(0)]


def test_results_empty(keyed: FakeArctend, capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run(["results"], capsys)
    assert code == 0 and out.strip() == cli.NO_RESULTS
    assert cli.NO_RESULTS == (
        "No practice results yet: a result appears shortly after its practice question closes."
    )


def test_results_outside_the_sandbox(
    keyed: FakeArctend, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    keyed.serves_practice = False
    code, out, err = run_main(["results"], capsys, monkeypatch)
    assert code == 1 and out == ""
    assert err.strip() == (
        "Practice results are served only by the practice sandbox (https://sandbox.arctend.com)."
    )


def test_results_without_a_key(
    fake: FakeArctend, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("ARCTEND_API_URL", fake.url)
    monkeypatch.delenv("ARCTEND_API_KEY", raising=False)
    code, _, err = run(["results"], capsys)
    assert code == 2 and "ARCTEND_API_KEY" in err and fake.requests == []


# --- the console command ------------------------------------------------------------------------


def test_the_console_command_is_declared(capsys: pytest.CaptureFixture[str]) -> None:
    scripts = importlib.metadata.entry_points(group="console_scripts")
    (entry,) = [ep for ep in scripts if ep.name == "arctend"]
    assert entry.value == "arctend.__main__:main"
    with pytest.raises(SystemExit):
        cli.run(["--help"])
    help_text = capsys.readouterr().out
    assert help_text.startswith("usage: arctend")
    for command in (
        "register",
        "check",
        "questions",
        "question",
        "consensus",
        "me",
        "rating",
        "forecast",
        "results",
        "run",
    ):
        assert command in help_text


# --- network failures ---------------------------------------------------------------------------


def _raising_opener(error: Exception) -> Callable[[object, float], object]:
    def opener(request: object, timeout: float) -> object:
        raise error

    return opener


def test_a_certificate_failure_is_one_line(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cert = ssl.SSLCertVerificationError(
        1, "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed"
    )
    monkeypatch.setattr(api, "open_without_redirects", _raising_opener(urllib.error.URLError(cert)))
    monkeypatch.delenv("ARCTEND_API_URL", raising=False)
    code, _, err = run_main(["questions"], capsys, monkeypatch)
    assert code == 1
    assert err.strip() == (
        "error: could not verify the certificate of sandbox.arctend.com. On macOS with Python "
        'from python.org, run "Install Certificates.command" in the Python folder under '
        "Applications; behind a company proxy, set SSL_CERT_FILE to its certificate bundle."
    )
    assert "Traceback" not in err


def test_a_connection_failure_is_one_line(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    refused = urllib.error.URLError(ConnectionRefusedError(10061, "connection refused"))
    monkeypatch.setattr(api, "open_without_redirects", _raising_opener(refused))
    monkeypatch.setenv("ARCTEND_API_URL", "https://api.arctend.com")
    monkeypatch.setenv("ARCTEND_API_KEY", "atk_x")
    code, out, err = run_main(["forecast", "q1", "--probability", "0.5"], capsys, monkeypatch)
    assert code == 1 and out.startswith("API: https://api.arctend.com\n")
    assert err.startswith("error: could not reach api.arctend.com: ")
    assert "connection refused" in err and len(err.strip().splitlines()) == 1
    assert "Traceback" not in err


# --- console encoding ---------------------------------------------------------------------------


def test_a_title_the_console_cannot_encode_does_not_crash(
    keyed: FakeArctend, tmp_path: Path
) -> None:
    keyed.add(make_question("binary", 1, title="Will 東京 see more than 30 mm of rain? Ωmega"))
    env = dict(
        os.environ,
        ARCTEND_API_URL=keyed.url,
        ARCTEND_API_KEY=keyed.api_key,
        PYTHONIOENCODING="cp1252",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "arctend", "questions"],
        capture_output=True,
        env=env,
        cwd=tmp_path,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
    out = proc.stdout.decode("cp1252")
    assert "Will ?? see more than 30 mm of rain? ?mega" in out
