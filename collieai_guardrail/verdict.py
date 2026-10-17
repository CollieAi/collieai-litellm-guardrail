"""What a reply means.

A 200 is a completed check only when every condition of the strict parse holds; anything else is the `protocol`
bypass, with the first condition that failed as its detail. A block or a rewrite in Monitor is a protocol error, not
a decision: the adapter never acts on a reply. A non-2xx is a bypass named by the contract's table from its status
and `error.type`; without a JSON body or without `error.type` it is `http_<class>`.

The diagnostics kept from a check are built field by field from what the contract defines — identifiers, closed
codes, counts, positions and timings — so nothing else a reply might carry reaches LiteLLM's logs.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeGuard

#: Keys a Monitor reply never carries, not even as null: stock generic applies `texts`, `tools` and
#: `structured_messages` on any action but `BLOCKED`.
REPLACEMENT_KEYS = (
    "texts",
    "images",
    "structured_messages",
    "tools",
    "tool_calls",
    "stream_holdback_chars",
    "blocked_reason",
)
COVERAGE_STATES = ("clean", "degraded")
#: What Enforce would have done to the call where a check found something (`would.outcome`).
FLAGGED_OUTCOMES = ("transform", "block")
#: The stages the server times; no other name is kept.
TIMINGS = ("normalize", "pin", "check", "judge", "total")

#: The server's error table for a Monitor reply: (status, error.type) → the bypass reason.
ERROR_REASONS: dict[tuple[int, str], str] = {
    (404, "not_found_error"): "not_found",
    (401, "authentication_error"): "auth",
    (403, "forbidden_error"): "auth",
    (413, "limit"): "limit",
    (422, "malformed"): "protocol",
    (422, "shape"): "shape",
    (422, "limit"): "limit",
    (422, "capability"): "capability",
    (422, "protocol"): "protocol",
    (503, "busy"): "busy",
    (503, "pin"): "unavailable",
    (504, "deadline"): "deadline",
    (500, "pin"): "server_error",
    (500, "execution"): "server_error",
    (500, "internal"): "server_error",
}
#: 429 keeps the application's existing envelope, so its status alone names it.
RATE_LIMITED_STATUS = 429

#: Every bypass reason the adapter counts.
BYPASS_REASONS = (
    "not_found",
    "auth",
    "rate_limited",
    "limit",
    "protocol",
    "shape",
    "capability",
    "busy",
    "unavailable",
    "deadline",
    "server_error",
    "http_1xx",
    "http_3xx",
    "http_4xx",
    "http_5xx",
    "timeout",
    "transport",
    "adapter_error",
    "cancelled",  # the caller's call was cancelled while the phase waited for the server
)

_SHORT = 256


@dataclass(frozen=True)
class Check:
    """A completed Monitor check: the server judged every item of the phase. `flagged` — Enforce would have masked
    or blocked: the call went on unchanged, but the check found something."""

    evaluation_id: str
    coverage_state: str
    would_outcome: str | None
    flagged: bool
    diagnostics: dict[str, object]


@dataclass(frozen=True)
class Rejected:
    """A 200 that is not a check. `detail` is one of a fixed set of codes."""

    detail: str


def parse_check(body: bytes, *, phase: str) -> Check | Rejected:
    try:
        reply = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return Rejected("not_json")
    if not isinstance(reply, dict):
        return Rejected("not_object")
    if reply.get("action") != "NONE":
        return Rejected("action")
    if any(key in reply for key in REPLACEMENT_KEYS):
        return Rejected("replacement_key")
    collieai = reply.get("collieai")
    if not isinstance(collieai, dict):
        return Rejected("diagnostics_missing")
    if not _is_int(collieai.get("version")) or collieai.get("version") != 1:
        return Rejected("version")
    if collieai.get("mode") != "monitor":
        return Rejected("mode")
    if collieai.get("phase") != phase:
        return Rejected("phase")
    if collieai.get("terminal") != "completed":
        return Rejected("terminal")
    coverage = collieai.get("coverage")
    if not isinstance(coverage, dict) or coverage.get("state") not in COVERAGE_STATES:
        return Rejected("coverage_state")
    judged, total, rules_run = coverage.get("items_judged"), coverage.get("items_total"), coverage.get("rules_run")
    if not (_is_int(judged) and _is_int(total) and judged == total and total >= 1):
        return Rejected("items")
    if not isinstance(rules_run, list) or len(rules_run) != total:
        return Rejected("rules_run")
    if not all(_is_int(count) and count >= 0 for count in rules_run):
        return Rejected("rules_run")
    evaluation_id = collieai.get("evaluation_id")
    if not isinstance(evaluation_id, str) or not evaluation_id or len(evaluation_id) > _SHORT:
        return Rejected("evaluation_id")
    diagnostics = _diagnostics(collieai)
    kept_would = diagnostics.get("would")
    would: dict[str, object] = kept_would if isinstance(kept_would, dict) else {}
    kept_outcome = would.get("outcome")
    outcome = kept_outcome if isinstance(kept_outcome, str) else None
    return Check(
        evaluation_id=evaluation_id,
        coverage_state=str(coverage["state"]),
        would_outcome=outcome,
        flagged=outcome in FLAGGED_OUTCOMES or would.get("would_block") is True or would.get("would_mask") is True,
        diagnostics=diagnostics,
    )


def error_reason(status: int, body: bytes) -> str:
    """The bypass reason of a reply that is not a 2xx."""
    if status == RATE_LIMITED_STATUS:
        return "rate_limited"
    error_type = _error_type(body)
    if error_type is not None and (status, error_type) in ERROR_REASONS:
        return ERROR_REASONS[(status, error_type)]
    return f"http_{status // 100}xx" if 100 <= status < 600 else "protocol"


def _error_type(body: bytes) -> str | None:
    try:
        reply = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return None
    error = reply.get("error") if isinstance(reply, dict) else None
    error_type = error.get("type") if isinstance(error, dict) else None
    return error_type if isinstance(error_type, str) else None


def _is_int(value: object) -> TypeGuard[int]:
    return type(value) is int


# ------------------------------------------------------------------ the diagnostics kept


def _diagnostics(collieai: Mapping[str, object]) -> dict[str, object]:
    out: dict[str, object] = {}
    for key in ("version", "evaluation_id", "phase", "terminal", "mode", "profile", "policy_fingerprint"):
        value = collieai.get(key)
        if _is_int(value) or _short_str(value):
            out[key] = value
    coverage = collieai.get("coverage")
    if isinstance(coverage, Mapping):
        out["coverage"] = {
            "state": coverage.get("state"),
            "items_judged": coverage.get("items_judged"),
            "items_total": coverage.get("items_total"),
            "rules_run": _ints(coverage.get("rules_run")),
            "degraded_rules": _strs(coverage.get("degraded_rules")),
        }
    would = collieai.get("would")
    if isinstance(would, Mapping):
        out["would"] = {
            "outcome": _str_or_none(would.get("outcome")),
            "complete": _bool_or_none(would.get("complete")),
            "approximate": _bool_or_none(would.get("approximate")),
            "would_block": _bool_or_none(would.get("would_block")),
            "would_mask": _bool_or_none(would.get("would_mask")),
            "masked_items": _ints(would.get("masked_items")),
        }
    items = collieai.get("items")
    if isinstance(items, list):
        out["items"] = [_item(item) for item in items if isinstance(item, Mapping)]
    timings = collieai.get("timings_ms")
    if isinstance(timings, Mapping):
        out["timings_ms"] = {
            key: value
            for key in TIMINGS
            if isinstance(value := timings.get(key), (int, float)) and not isinstance(value, bool)
        }
    return out


def _item(item: Mapping[str, object]) -> dict[str, object]:
    findings = item.get("findings")
    return {
        "index": item.get("index") if _is_int(item.get("index")) else None,
        "source": _str_or_none(item.get("source")),
        "outcome": _str_or_none(item.get("outcome")),
        "findings": [
            {key: _str_or_none(finding.get(key)) for key in ("rule_id", "rule_type", "action", "mode")}
            for finding in (findings if isinstance(findings, list) else [])
            if isinstance(finding, Mapping)
        ],
    }


def _short_str(value: object) -> bool:
    return isinstance(value, str) and len(value) <= _SHORT


def _str_or_none(value: object) -> str | None:
    return value if isinstance(value, str) and len(value) <= _SHORT else None


def _bool_or_none(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def _ints(value: object) -> list[int]:
    return [v for v in value if _is_int(v)] if isinstance(value, list) else []


def _strs(value: object) -> list[str]:
    return [v for v in value if _short_str(v)] if isinstance(value, list) else []
