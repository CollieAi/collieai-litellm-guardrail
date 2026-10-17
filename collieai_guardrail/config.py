"""The adapter's settings, as the operator gives them in the guardrail's `litellm_params`.

Two numbers bound a phase:

- `deadline_ms` — the adapter's own absolute deadline for one phase: the connection from the pool, the send, the
  reply read in full and its parse. Nothing the adapter does for a phase outlives it.
- `network_margin_ms` — the part of that deadline kept for the network. The server gets the rest as `budget_ms`;
  the server caps it at its own ceiling (2000 ms today), so a larger budget is not an error.

The defaults fit the server's ceiling.
"""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass, field
from urllib.parse import urlsplit

#: The route under the operator's `api_base`, e.g. `https://app.collieai.io/v1/integrations/litellm`.
ROUTE = "/beta/litellm_basic_guardrail_api"
DEFAULT_DEADLINE_MS = 2500
DEFAULT_NETWORK_MARGIN_MS = 500
#: A phase is never given more than a minute: the adapter runs inside the user's request.
MAX_DEADLINE_MS = 60_000
#: The server's body limit of the narrow profile: a request over it is answered 413 there, so the
#: adapter does not send it — the bytes would cost the user's request time for a refusal.
DEFAULT_MAX_BODY_BYTES = 448 * 1024
#: The largest reply read. A Monitor reply carries counts, closed codes and identifiers, never text.
MAX_REPLY_BYTES = 1024 * 1024
#: How often each worker logs its counts, and how often it runs the synthetic check (0: never).
DEFAULT_SUMMARY_INTERVAL_S = 60
DEFAULT_SYNTHETIC_INTERVAL_S = 300


#: Every CollieAi project key starts with it. An unset `os.environ/COLLIEAI_API_KEY` reaches the
#: adapter as None from the proxy's config, or as the string "None" from LiteLLM's guardrail registry, which resolves
#: the variable itself for a guardrail added through its API; a key without the prefix is taken for a missing one.
KEY_PREFIX = "clai_"


class ConfigError(Exception):
    """A setting the adapter cannot run with. The message names the setting, never its value.

    Not a ValueError or a TypeError: LiteLLM's loader of the config's guardrails (`init_guardrails_v2`) catches those,
    logs one ERROR and starts the proxy without the guardrail. This one stops the proxy's start, so nothing that
    checks a setting may let a ValueError or a TypeError out."""


@dataclass(frozen=True)
class AdapterConfig:
    api_base: str
    api_key: str = field(repr=False)
    deadline_ms: int = DEFAULT_DEADLINE_MS
    network_margin_ms: int = DEFAULT_NETWORK_MARGIN_MS
    max_body_bytes: int = DEFAULT_MAX_BODY_BYTES
    #: Forward the caller's `x-collieai-conversation-id` header, and only that header, to the server.
    forward_conversation_id: bool = False
    #: The worker's counts go to the log at INFO this often.
    summary_interval_s: int = DEFAULT_SUMMARY_INTERVAL_S
    #: A synthetic pre- and post-call observation runs at start and then this often; 0 turns it off.
    synthetic_interval_s: int = DEFAULT_SYNTHETIC_INTERVAL_S

    def __post_init__(self) -> None:
        if not isinstance(self.api_base, str):
            raise ConfigError("api_base must be a URL")
        try:
            parts = urlsplit(self.api_base)
            port = parts.port  # a port that is not a number from 0 to 65535 raises here, not at the first call
        except ValueError:  # its message may carry a part of the URL
            raise ConfigError("api_base must be a URL") from None
        if parts.scheme not in ("http", "https") or not parts.hostname or port == 0:
            raise ConfigError("api_base must be an http(s) URL with a host")
        if parts.scheme == "http" and not _is_loopback(parts.hostname):
            # the key and the caller's text would cross the network in clear: https, or a stand on this machine
            raise ConfigError("api_base must use https; http only for a loopback address (127.0.0.1, ::1, localhost)")
        if parts.username is not None or parts.password is not None or parts.query or parts.fragment:
            raise ConfigError("api_base must not carry credentials, a query or a fragment")
        if not isinstance(self.api_key, str) or not self.api_key or self.api_key != self.api_key.strip():
            raise ConfigError("api_key must be a non-empty string without surrounding whitespace")
        if any(ch in self.api_key for ch in "\r\n\x00"):
            raise ConfigError("api_key must be one line")
        if not self.api_key.startswith(KEY_PREFIX) or self.api_key == KEY_PREFIX:
            raise ConfigError(f"api_key must be a CollieAi project key ({KEY_PREFIX}…); is its variable set?")
        _int_in("deadline_ms", self.deadline_ms, 1, MAX_DEADLINE_MS)
        _int_in("network_margin_ms", self.network_margin_ms, 0, self.deadline_ms - 1)
        _int_in("max_body_bytes", self.max_body_bytes, 1, 64 * 1024 * 1024)
        if not isinstance(self.forward_conversation_id, bool):
            raise ConfigError("forward_conversation_id must be true or false")
        _int_in("summary_interval_s", self.summary_interval_s, 1, 3600)
        _int_in("synthetic_interval_s", self.synthetic_interval_s, 0, 86_400)

    @property
    def url(self) -> str:
        return self.api_base.rstrip("/") + ROUTE

    @property
    def budget_ms(self) -> int:
        """The server's share of the deadline: what remains after the network's margin, at least 1 ms."""
        return self.deadline_ms - self.network_margin_ms


def _is_loopback(host: str) -> bool:
    """`localhost`, or an IP literal on the loopback network (127.0.0.0/8, ::1)."""
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:  # a name other than localhost: it may resolve anywhere
        return False


def _int_in(name: str, value: object, low: int, high: int) -> None:
    if type(value) is not int or not low <= value <= high:
        raise ConfigError(f"{name} must be an integer from {low} to {high}")
