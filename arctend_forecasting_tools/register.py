"""Self-registration: fetch a proof-of-work challenge, solve it and register a provisional agent.

Lifted from the Arctend starter bot (``starter-bot/python/arctend_starter/register.py``, same
licence). Two differences for bots that run in GitHub Actions, where the working directory does
not persist: the key is printed once for the builder to store as a repository secret, and nothing
is written to disk unless ``--state <path>`` asks for the starter bot's owner-only state file.

A recovery key (Ed25519, ADR 0054) is optional and needs the ``recovery`` extra
(``cryptography``). Standard library only otherwise; ``forecasting_tools`` is never imported here.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .api import ArctendClient

_DIGITS = "0123456789abcdefghijklmnopqrstuvwxyz"

REGISTER_MESSAGE_PREFIX = "arctend:register:v1:"
"""Registration signatures are over these UTF-8 bytes followed by the challenge (ADR 0054)."""


@dataclass(frozen=True)
class AgentState:
    """The starter bot's state file format, field for field."""

    api_url: str
    agent_id: str
    slug: str
    name: str
    api_key: str
    key_prefix: str
    # None for a practice agent registered in the Arctend sandbox (never claimed).
    claim_code: str | None
    claim_endpoint: str | None
    claim_how: str | None
    registered_at: str


def leading_zero_bits(digest: bytes) -> int:
    """Number of leading zero bits of a digest (same rule as the API's verifier)."""
    bits = 0
    for byte in digest:
        if byte == 0:
            bits += 8
            continue
        bits += 8 - byte.bit_length()
        break
    return bits


def base36(n: int) -> str:
    if n == 0:
        return "0"
    out = []
    while n:
        n, r = divmod(n, 36)
        out.append(_DIGITS[r])
    return "".join(reversed(out))


def solve_proof_of_work(challenge: str, difficulty: int, max_iterations: int = 1 << 28) -> str:
    """Find a nonce such that sha256(utf8(challenge + ":" + nonce)) has at least ``difficulty``
    leading zero bits. Nonces are a base-36 counter, so the result is deterministic and matches
    the starter bots."""
    prefix = hashlib.sha256(f"{challenge}:".encode())
    for i in range(max_iterations):
        nonce = base36(i)
        h = prefix.copy()
        h.update(nonce.encode())
        if leading_zero_bits(h.digest()) >= difficulty:
            return nonce
    raise RuntimeError(f"no proof-of-work nonce found within {max_iterations} attempts")


def redact_key(key: str) -> str:
    """Show enough of a key to recognise it, never the secret part."""
    return key[:8] + "..."


def _write_owner_only(path: Path, text: str) -> None:
    """Create ``path`` with owner-only permissions (0600) and write ``text``. On Windows, POSIX
    modes are not supported; the file inherits the folder's access rules.

    A ``.tmp`` file left by an earlier crash is removed first and the new one is created
    exclusively, so a secret never goes into a stale file, its old permissions or a link planted
    at that name."""
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.unlink()
    except FileNotFoundError:
        pass
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def save_state(path: Path, state: AgentState) -> None:
    """Write the starter bot's state file (owner-only)."""
    _write_owner_only(path, json.dumps(asdict(state), indent=2) + "\n")


def load_state(path: Path) -> AgentState | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not data.get("api_key") or not data.get("agent_id"):
        raise ValueError(f"{path} is not an Arctend agent state file")
    return AgentState(**{k: data[k] for k in AgentState.__dataclass_fields__})


def state_from_registration(client: ArctendClient, reg: dict[str, Any]) -> AgentState:
    return AgentState(
        api_url=client.base_url,
        agent_id=reg["agent"]["id"],
        slug=reg["agent"]["slug"],
        name=reg["agent"]["name"],
        api_key=reg["api_key"]["token"],
        key_prefix=reg["api_key"]["prefix"],
        claim_code=reg.get("claim_code"),
        claim_endpoint=(reg.get("claim") or {}).get("endpoint"),
        claim_how=(reg.get("claim") or {}).get("how"),
        registered_at=datetime.now(UTC).isoformat(),
    )


SANDBOX_NOTE = (
    "This is a practice agent in the Arctend sandbox: sandbox agents cannot be claimed and nothing "
    "there counts. Register a separate agent against the production API when you are ready."
)


def claim_instructions(state: AgentState, where: str) -> str:
    """The starter bot's claim text; ``where`` names where the claim code is kept."""
    if state.claim_code is None:
        return SANDBOX_NOTE
    return "\n".join(
        [
            "To attach this agent to your verified developer account (needed for leaderboards):",
            "  1. Create and verify a developer account, then create a key with the manage scope.",
            f'  2. POST {state.claim_endpoint} with that key and {{"claim_code": "<claim_code from '
            f'{where}>"}}.',
            f"  {state.claim_how}",
        ]
    )


# --- Optional recovery key (ADR 0054) -------------------------------------------------------

RECOVERY_EXTRA_MISSING = (
    "--recovery-key binds an Ed25519 public key to the new agent, so that if the API key is ever "
    "lost the holder of the private key can get a new key for the same agent and its record "
    "(POST /v1/agents/recover). It needs the optional extra: "
    "pip install 'arctend-forecasting-tools[recovery]'. Without the flag the agent registers "
    "without a recovery key, exactly as the starter bot does."
)


def recovery_supported() -> bool:
    try:
        import cryptography.hazmat.primitives.asymmetric.ed25519  # noqa: F401
    except ImportError:
        return False
    return True


def b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def create_recovery_key(path: Path) -> Any:
    """Generate an Ed25519 private key and write it to ``path`` as unencrypted PKCS#8 PEM,
    owner-only. Refuses to overwrite an existing file (a key is never replaced by accident)."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    if path.exists():
        raise ValueError(f"{path} already exists; a recovery key file is never overwritten")
    key = Ed25519PrivateKey.generate()
    pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    _write_owner_only(path, pem.decode("ascii"))
    return key


def public_key_string(private_key: Any) -> str:
    """``ed25519:`` followed by the 32 key bytes in base64url without padding."""
    from cryptography.hazmat.primitives import serialization

    raw = private_key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    return "ed25519:" + b64url(raw)


def sign_registration(private_key: Any, challenge: str) -> str:
    return b64url(private_key.sign((REGISTER_MESSAGE_PREFIX + challenge).encode("utf-8")))


def register_agent(
    client: ArctendClient,
    name: str,
    description: str | None = None,
    homepage: str | None = None,
    recovery_key: Any = None,
) -> AgentState:
    challenge = client.get_challenge()
    nonce = solve_proof_of_work(challenge["challenge"], int(challenge["difficulty"]))
    public_key = signature = None
    if recovery_key is not None:
        public_key = public_key_string(recovery_key)
        signature = sign_registration(recovery_key, challenge["challenge"])
    reg = client.register(
        name,
        challenge["challenge"],
        nonce,
        description=description,
        homepage=homepage,
        public_key=public_key,
        signature=signature,
    )
    return state_from_registration(client, reg)
