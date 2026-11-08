"""Acceptance tests 5 to 7: the runner end to end against the fake Arctend server."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from conftest import ExplodingMetaculusClient, FakeArctend, StubBot, make_question

from arctend_forecasting_tools import ArctendClient, forecast_on_arctend
from arctend_forecasting_tools.runner import ArctendStopError


def client_for(fake: FakeArctend, sleeps: list[float] | None = None, key: str | None = None):
    record = sleeps if sleeps is not None else []
    return ArctendClient(fake.url, key or fake.api_key, sleep=record.append)


def today() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")


def three_questions(fake: FakeArctend) -> tuple[dict, dict, dict]:
    b = fake.add(make_question("binary", 1))
    m = fake.add(make_question("multiple_choice", 2))
    n = fake.add(make_question("numeric", 3), standing_days_ago=2)
    return b, m, n


# --- 5. End to end ------------------------------------------------------------------------------


async def test_end_to_end_skip_submit_and_same_day_rerun(
    fake: FakeArctend, capsys: pytest.CaptureFixture[str]
) -> None:
    b, m, n = three_questions(fake)
    bot = StubBot(skip_previously_forecasted_questions=True)
    results = await forecast_on_arctend(bot, client_for(fake), refresh_days=7)

    by_id = {r.question_id: r for r in results}
    assert by_id[b["id"]].status == "submitted"
    assert by_id[m["id"]].status == "submitted"
    assert by_id[n["id"]].status == "skipped"
    assert sorted(r.status for r in results) == ["skipped", "submitted", "submitted"]

    puts = fake.puts()
    assert len(puts) == 2
    keys = {p["path"].split("/")[3]: p["headers"]["Idempotency-Key"] for p in puts}
    assert keys == {
        b["id"]: f"arctend-ft:{b['id']}:{today()}",
        m["id"]: f"arctend-ft:{m['id']}:{today()}",
    }
    bodies = {p["path"].split("/")[3]: p["body"] for p in puts}
    assert bodies[b["id"]]["probability"] == 0.62
    assert bodies[m["id"]]["vector"] == pytest.approx([0.5, 0.3, 0.2])
    for body in bodies.values():
        assert body["agent_version"].startswith("StubBot via arctend-forecasting-tools/")
        assert body["rationale"]
    assert by_id[b["id"]].receipt["forecast_id"]
    # The skipped question was never fetched in full and never shown to the bot.
    assert f"/v1/questions/{n['id']}" not in fake.paths()
    assert [q.question_text for q in bot.seen] == [b["title"], m["title"]]
    assert "tier provisional, 200 forecasts per day" in capsys.readouterr().out

    again = await forecast_on_arctend(bot, client_for(fake), refresh_days=7)
    assert len(fake.puts()) == 2  # no PUT on the second run the same day
    assert [r.status for r in again] == ["skipped", "skipped", "skipped"]


async def test_skip_flag_off_reforecasts_and_marks_already_forecasted(fake: FakeArctend) -> None:
    fake.add(make_question("binary", 1), standing_days_ago=2)
    bot = StubBot(skip_previously_forecasted_questions=False)
    results = await forecast_on_arctend(bot, client_for(fake), refresh_days=7)
    assert [r.status for r in results] == ["submitted"]
    assert bot.seen[0].already_forecasted is True


async def test_old_forecast_is_not_fresh(fake: FakeArctend) -> None:
    fake.add(make_question("binary", 1), standing_days_ago=9)
    bot = StubBot(skip_previously_forecasted_questions=True)
    results = await forecast_on_arctend(bot, client_for(fake), refresh_days=7)
    assert [r.status for r in results] == ["submitted"]
    assert bot.seen[0].already_forecasted is False


async def test_max_questions_caps_after_the_skip(fake: FakeArctend) -> None:
    fake.add(make_question("binary", 1), standing_days_ago=1)
    for n in (2, 3, 4):
        fake.add(make_question("binary", n))
    bot = StubBot(skip_previously_forecasted_questions=True)
    results = await forecast_on_arctend(bot, client_for(fake), max_questions=2)
    assert [r.status for r in results] == ["skipped", "submitted", "submitted"]
    assert len(fake.puts()) == 2


async def test_dry_run_submits_nothing(fake: FakeArctend, capsys: pytest.CaptureFixture) -> None:
    three_questions(fake)
    bot = StubBot(skip_previously_forecasted_questions=True)
    results = await forecast_on_arctend(bot, client_for(fake), dry_run=True)
    assert sorted(r.status for r in results) == ["dry_run", "dry_run", "skipped"]
    assert fake.puts() == []
    assert len(bot.seen) == 2  # the bot still ran
    assert "dry-run question-1: {'probability': 0.62" in capsys.readouterr().out


async def test_no_rationale_and_custom_agent_version(fake: FakeArctend) -> None:
    fake.add(make_question("binary", 1))
    await forecast_on_arctend(
        StubBot(), client_for(fake), include_rationale=False, agent_version="mine-1"
    )
    assert fake.puts()[0]["body"] == {"probability": 0.62, "agent_version": "mine-1"}


async def test_closed_questions_are_ignored(fake: FakeArctend) -> None:
    fake.add(make_question("binary", 1, close_at="2020-01-01T00:00:00.000Z"))
    results = await forecast_on_arctend(StubBot(), client_for(fake))
    assert results == [] and fake.puts() == []


# --- 6. Publication to Metaculus forced off -----------------------------------------------------


async def test_publication_is_forced_off_and_restored(
    fake: FakeArctend, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("METACULUS_TOKEN", raising=False)
    three_questions(fake)
    ExplodingMetaculusClient.touched = False
    bot = StubBot(
        publish_reports_to_metaculus=True,
        metaculus_client=ExplodingMetaculusClient(),
        skip_previously_forecasted_questions=True,
    )
    results = await forecast_on_arctend(bot, client_for(fake))
    assert sorted(r.status for r in results) == ["skipped", "submitted", "submitted"]
    assert ExplodingMetaculusClient.touched is False
    assert bot.publish_reports_to_metaculus is True


async def test_publication_restored_when_the_bot_raises(fake: FakeArctend) -> None:
    fake.add(make_question("binary", 1))
    seen_flag: list[bool] = []

    class Raising(StubBot):
        async def forecast_questions(self, questions: Any, return_exceptions: bool = False):
            seen_flag.append(self.publish_reports_to_metaculus)
            raise RuntimeError("bot crashed mid-run")

    bot = Raising(publish_reports_to_metaculus=True, metaculus_client=ExplodingMetaculusClient())
    with pytest.raises(RuntimeError, match="mid-run"):
        await forecast_on_arctend(bot, client_for(fake))
    assert seen_flag == [False]
    assert bot.publish_reports_to_metaculus is True
    assert fake.puts() == []


# --- 7. Errors ----------------------------------------------------------------------------------


async def test_bot_exception_on_one_question(fake: FakeArctend) -> None:
    b, m, n = (fake.add(make_question("binary", i)) for i in (1, 2, 3))
    bot = StubBot(fail_titles=(m["title"],))
    results = await forecast_on_arctend(bot, client_for(fake))
    status = {r.question_id: r for r in results}
    assert status[m["id"]].status == "error"
    assert "research failed" in status[m["id"]].error
    assert status[b["id"]].status == "submitted" and status[n["id"]].status == "submitted"
    assert len(fake.puts()) == 2


async def test_mapping_error_is_reported_and_others_continue(fake: FakeArctend) -> None:
    bad = fake.add(
        make_question(
            "multiple_choice", 1, options=[{"idx": 0, "label": "A"}, {"idx": 1, "label": "A"}]
        )
    )
    ok = fake.add(make_question("binary", 2))
    results = await forecast_on_arctend(StubBot(), client_for(fake))
    status = {r.question_id: r.status for r in results}
    assert status == {bad["id"]: "error", ok["id"]: "submitted"}


async def test_429_is_retried_once(fake: FakeArctend) -> None:
    b = fake.add(make_question("binary", 1))
    path = f"/v1/questions/{b['id']}/forecast"
    fake.script[("PUT", path)] = [(429, {"Retry-After": "1"}, {"code": "rate_limited"})]
    sleeps: list[float] = []
    results = await forecast_on_arctend(StubBot(), client_for(fake, sleeps))
    assert [r.status for r in results] == ["submitted"]
    assert sleeps == [1]
    assert len(fake.puts()) == 2


async def test_second_429_stops_the_run(fake: FakeArctend) -> None:
    qs = [fake.add(make_question("binary", i)) for i in (1, 2, 3)]
    path = f"/v1/questions/{qs[0]['id']}/forecast"
    limited = (429, {"Retry-After": "1"}, {"code": "rate_limited"})
    fake.script[("PUT", path)] = [limited, limited]
    sleeps: list[float] = []
    results = await forecast_on_arctend(StubBot(), client_for(fake, sleeps))
    assert [r.status for r in results] == ["not_submitted"] * 3
    assert sleeps == [1]
    assert len(fake.puts()) == 2  # the first question twice, nothing after


async def test_401_on_me_stops_before_any_question(fake: FakeArctend) -> None:
    fake.add(make_question("binary", 1))
    bot = StubBot()
    with pytest.raises(ArctendStopError, match="rejected"):
        await forecast_on_arctend(bot, client_for(fake, key="atk_wrong.key"))
    assert fake.paths() == ["/v1/me"]
    assert bot.seen == []


async def test_missing_key_stops_without_a_request(fake: FakeArctend) -> None:
    with pytest.raises(ArctendStopError, match="ARCTEND_API_KEY"):
        await forecast_on_arctend(StubBot(), ArctendClient(fake.url))
    assert fake.requests == []


async def test_conflict_on_one_put_does_not_stop_the_run(fake: FakeArctend) -> None:
    a, b = fake.add(make_question("binary", 1)), fake.add(make_question("binary", 2))
    fake.script[("PUT", f"/v1/questions/{a['id']}/forecast")] = [
        (409, {}, {"code": "idempotency_conflict", "detail": "different body"})
    ]
    results = await forecast_on_arctend(StubBot(), client_for(fake))
    status = {r.question_id: r.status for r in results}
    assert status == {a["id"]: "error", b["id"]: "submitted"}
