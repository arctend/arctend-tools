"""Acceptance test 8: ``register`` against the fake server, with and without the extras."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import stat
import sys
from pathlib import Path

import pytest
from conftest import FakeArctend

from arctend_forecasting_tools import __main__ as cli
from arctend_forecasting_tools.register import (
    AgentState,
    base36,
    leading_zero_bits,
    load_state,
    save_state,
    solve_proof_of_work,
)

KEY = "atk_fresh_secret_token_123456"
CLAIM = "claim-code-secret-987"

# The starter bot's state file fields, in order (starter-bot/python/arctend_starter/register.py).
STARTER_FIELDS = [
    "api_url",
    "agent_id",
    "slug",
    "name",
    "api_key",
    "key_prefix",
    "claim_code",
    "claim_endpoint",
    "claim_how",
    "registered_at",
]


@pytest.fixture
def env(fake: FakeArctend, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("ARCTEND_API_URL", fake.url)
    monkeypatch.delenv("ARCTEND_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def registration_body(fake: FakeArctend) -> dict:
    posts = [r for r in fake.requests if r["method"] == "POST"]
    assert len(posts) == 1
    return posts[0]["body"]


def test_register_prints_key_and_claim_code_once_and_writes_nothing(
    fake: FakeArctend, env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = cli.run(["register", "--name", "My bot", "--description", "Template bot"])
    out = capsys.readouterr().out
    assert code == 0
    assert out.count(KEY) == 1
    assert out.count(CLAIM) == 1
    assert "POST " + fake.url + "/v1/agents/claim" in out
    assert list(env.iterdir()) == []
    body = registration_body(fake)
    assert body["name"] == "My bot" and body["description"] == "Template bot"
    assert "public_key" not in body and "signature" not in body
    assert "homepage" not in body


def test_register_in_the_sandbox_prints_the_sandbox_note(
    fake: FakeArctend, env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake.sandbox = True
    assert cli.run(["register", "--name", "Practice bot"]) == 0
    out = capsys.readouterr().out
    assert out.count(KEY) == 1
    assert "Claim code" not in out
    assert "sandbox agents cannot be claimed" in out


def test_register_with_state_writes_the_starter_format(
    fake: FakeArctend, env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = env / "agent.json"
    assert cli.run(["register", "--name", "My bot", "--state", str(path)]) == 0
    data = json.loads(path.read_text(encoding="utf-8"))
    assert list(data) == STARTER_FIELDS
    assert data["api_key"] == KEY and data["claim_code"] == CLAIM
    assert data["api_url"] == fake.url
    assert load_state(path).api_key == KEY
    if sys.platform != "win32":
        assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    assert str(path) in capsys.readouterr().out
    # An existing state file is never overwritten.
    assert cli.run(["register", "--name", "Other", "--state", str(path)]) == 2
    assert json.loads(path.read_text(encoding="utf-8"))["api_key"] == KEY


def test_recovery_key_without_the_extra_exits_2(
    fake: FakeArctend,
    env: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    for name in list(sys.modules):
        if name == "cryptography" or name.startswith("cryptography."):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "cryptography", None)
    code = cli.run(["register", "--name", "My bot", "--recovery-key", str(env / "k.pem")])
    err = capsys.readouterr().err
    assert code == 2
    assert "arctend-forecasting-tools[recovery]" in err
    assert "lost" in err
    assert fake.requests == []
    assert not (env / "k.pem").exists()


def test_recovery_key_with_the_extra_binds_a_well_formed_key(
    fake: FakeArctend, env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    key_path = env / "recovery.pem"
    assert cli.run(["register", "--name", "My bot", "--recovery-key", str(key_path)]) == 0
    body = registration_body(fake)
    assert re.fullmatch(r"ed25519:[A-Za-z0-9_-]{43}", body["public_key"])
    assert re.fullmatch(r"[A-Za-z0-9_-]{86}", body["signature"])
    private = serialization.load_pem_private_key(key_path.read_bytes(), password=None)
    raw = base64.urlsafe_b64decode(body["public_key"].removeprefix("ed25519:") + "=")
    assert raw == private.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    sig = base64.urlsafe_b64decode(body["signature"] + "==")
    Ed25519PublicKey.from_public_bytes(raw).verify(sig, b"arctend:register:v1:test-challenge")
    if sys.platform != "win32":
        assert stat.S_IMODE(os.stat(key_path).st_mode) == 0o600
    out = capsys.readouterr().out
    assert out.count(KEY) == 1
    # An existing key file is never overwritten.
    before = key_path.read_bytes()
    assert cli.run(["register", "--name", "Again", "--recovery-key", str(key_path)]) == 2
    assert key_path.read_bytes() == before


def test_register_solves_the_proof_of_work(fake: FakeArctend, env: Path) -> None:
    fake.difficulty = 10
    assert cli.run(["register", "--name", "My bot"]) == 0
    body = registration_body(fake)
    digest = hashlib.sha256(f"{body['challenge']}:{body['nonce']}".encode()).digest()
    assert leading_zero_bits(digest) >= 10


# Vectors from the API's reference solver, as in the starter bots' tests.
@pytest.mark.parametrize(
    ("challenge", "difficulty", "nonce"),
    [
        ("starter-bot-vector-1", 8, "my"),
        ("starter-bot-vector-2", 12, "507"),
        ("eyJ.challenge.token", 16, "5iig"),
    ],
)
def test_pow_matches_api_reference_solver(challenge: str, difficulty: int, nonce: str) -> None:
    assert solve_proof_of_work(challenge, difficulty) == nonce


def test_base36_and_state_round_trip(tmp_path: Path) -> None:
    assert [base36(n) for n in (0, 35, 36, 1295)] == ["0", "z", "10", "zz"]
    state = AgentState(
        api_url="https://sandbox.arctend.com",
        agent_id="a",
        slug="s",
        name="n",
        api_key="atk_sbx_x",
        key_prefix="atk_sbx_",
        claim_code=None,
        claim_endpoint=None,
        claim_how=None,
        registered_at="2026-11-05T00:00:00+00:00",
    )
    save_state(tmp_path / "s.json", state)
    assert load_state(tmp_path / "s.json") == state
