"""What a reply means.

A 200 is a completed check only when every condition of the strict parse holds; anything else is the `protocol`
bypass, with the first condition that failed as its detail. A block or a rewrite in Monitor is a protocol error, not
a decision: in Monitor the adapter never acts on a reply. In `block` (enforce with blocking only) a reply is `NONE`
or `BLOCKED` from a server in `enforce` — a server that answers `monitor` would quietly lower the protection, so that
is a protocol error too — and the adapter blocks only on the top-level `BLOCKED`, never on what a finding or `would`
says. A non-2xx is a bypass named by the contract's table from its status and `error.type`; without a JSON body or
without `error.type` it is `http_<class>`. Its detail is the server's `error.code`, and the rule a refusal names goes
with it (`error_detail`).

A check found something when one of its findings is a `detection` — the server names each finding's `kind`: a
rule of the policy found what it looks for. A rule's `allow` and a detector that did
not run are no violation. The kind is all or nothing: a server before kinds names none, and then what Enforce would
have done decides, as it did; some named and some not, or a word this adapter does not know, is not a check.

The diagnostics kept from a check are built field by field from what the contract defines — identifiers, closed
codes, counts, positions and timings — so nothing else a reply might carry reaches LiteLLM's logs.
"""

from __future__ import annotations

import json
import re
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
#: A block stops the batch at the item that blocked, so the items after it are not judged.
BLOCKED_COVERAGE_STATES = ("clean", "degraded", "incomplete")
#: What Enforce would have done to the call where a check found something (`would.outcome`).
FLAGGED_OUTCOMES = ("transform", "block")
#: The stages the server times; no other name is kept.
TIMINGS = ("normalize", "pin", "check", "judge", "total")
#: A finding's `kind`: what the rule found (`detection`), a rule's `allow`, or a detector that did not run.
FINDING_KINDS = ("detection", "allow", "operational_failure")
DETECTION = "detection"
#: What `violation_categories` may name: CollieAi's rule types that check a message's text. A type outside it is
#: OTHER_CATEGORY — a word of the adapter, never one of the server's. So a check names at most all of them and
#: `other`, each at most 32 characters.
CATEGORIES = frozenset(
    {
        "aho_corasick",
        "base64_payload",
        "language_detection",
        "lightweight_model",
        "llm_detection",
        "normalization",
        "qwen3guard_stream",
        "regex",
        "structured_id",
        "url_filter",
        "url_reputation",
    }
)
OTHER_CATEGORY = "other"

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
#: The forms of what an error's detail takes from the server: a closed code, and a rule's ID. Anything else is left out.
_CODE = re.compile(r"[a-z][a-z0-9_]{0,63}")
_RULE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")


@dataclass(frozen=True)
class Check:
    """A completed Monitor check: the server judged every item of the phase. `flagged` — the check found something:
    the call went on unchanged. `categories` — the rule types of what it found, for `violation_categories`: from
    CATEGORIES or `other`, unique, sorted; none from a server before kinds, which does not say what a finding is."""

    evaluation_id: str
    coverage_state: str
    would_outcome: str | None
    flagged: bool
    diagnostics: dict[str, object]
    categories: tuple[str, ...] = ()
    #: Enforce only: the server blocked the call — the top-level `BLOCKED` — and the rule that did.
    blocked: bool = False
    blocked_by: str | None = None


@dataclass(frozen=True)
class Rejected:
    """A 200 that is not a check. `detail` is one of a fixed set of codes."""

    detail: str


def parse_check(body: bytes, *, phase: str, enforce: bool = False) -> Check | Rejected:
    """`enforce` — the adapter asked for `block`."""
    try:
        reply = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return Rejected("not_json")
    if not isinstance(reply, dict):
        return Rejected("not_object")
    action = reply.get("action")
    if action not in (("NONE", "BLOCKED") if enforce else ("NONE",)):
        return Rejected("action")
    blocked = action == "BLOCKED"
    if any(key in reply for key in REPLACEMENT_KEYS if not (blocked and key == "blocked_reason")):
        return Rejected("replacement_key")
    collieai = reply.get("collieai")
    if not isinstance(collieai, dict):
        return Rejected("diagnostics_missing")
    if not _is_int(collieai.get("version")) or collieai.get("version") != 1:
        return Rejected("version")
    if collieai.get("mode") != ("enforce" if enforce else "monitor"):
        return Rejected("mode")
    if collieai.get("phase") != phase:
        return Rejected("phase")
    if collieai.get("terminal") != ("blocked" if blocked else "completed"):
        return Rejected("terminal")
    coverage = collieai.get("coverage")
    states = BLOCKED_COVERAGE_STATES if blocked else COVERAGE_STATES
    if not isinstance(coverage, dict) or coverage.get("state") not in states:
        return Rejected("coverage_state")
    judged, total, rules_run = coverage.get("items_judged"), coverage.get("items_total"), coverage.get("rules_run")
    if not (
        _is_int(judged) and _is_int(total) and total >= 1 and (1 <= judged <= total if blocked else judged == total)
    ):
        return Rejected("items")
    if not isinstance(rules_run, list) or len(rules_run) != total:
        return Rejected("rules_run")
    if not all(_is_int(count) and count >= 0 for count in rules_run):
        return Rejected("rules_run")
    evaluation_id = collieai.get("evaluation_id")
    if not isinstance(evaluation_id, str) or not evaluation_id or len(evaluation_id) > _SHORT:
        return Rejected("evaluation_id")
    findings = _findings(collieai.get("items"))
    named = ["kind" in finding for finding in findings]
    if (any(named) or enforce) and not (all(named) and all(finding["kind"] in FINDING_KINDS for finding in findings)):
        return Rejected("finding_kind")
    blocked_by: str | None = None
    if blocked:
        acting = [f for f in findings if f.get("action") == "block" and f.get("mode") == "enforce"]
        if not acting:
            return Rejected("block_finding")
        blocked_by = _str_or_none(acting[0].get("rule_id"))
    diagnostics = _diagnostics(collieai)
    kept_would = diagnostics.get("would")
    would: dict[str, object] = kept_would if isinstance(kept_would, dict) else {}
    kept_outcome = would.get("outcome")
    outcome = kept_outcome if isinstance(kept_outcome, str) else None
    if findings and not any(named):  # a server before kinds: what Enforce would have done
        flagged = outcome in FLAGGED_OUTCOMES or would.get("would_block") is True or would.get("would_mask") is True
        categories: tuple[str, ...] = ()
    else:
        detections = [finding for finding in findings if finding["kind"] == DETECTION]
        flagged = bool(detections)
        categories = tuple(sorted({_category(finding.get("rule_type")) for finding in detections}))
    return Check(
        evaluation_id=evaluation_id,
        coverage_state=str(coverage["state"]),
        would_outcome=outcome,
        flagged=flagged,
        diagnostics=diagnostics,
        categories=categories,
        blocked=blocked,
        blocked_by=blocked_by,
    )


def error_reason(status: int, body: bytes) -> str:
    """The bypass reason of a reply that is not a 2xx."""
    if status == RATE_LIMITED_STATUS:
        return "rate_limited"
    error_type = _error(_object(body)).get("type")
    if isinstance(error_type, str) and (status, error_type) in ERROR_REASONS:
        return ERROR_REASONS[(status, error_type)]
    return f"http_{status // 100}xx" if 100 <= status < 600 else "protocol"


def error_detail(body: bytes) -> tuple[str | None, str | None]:
    """What a reply that is not a 2xx says of itself: its `error.code` — why, e.g. `aho_source` of a `capability`
    refusal — and the rule its refusal names (`collieai.refusal.rule_id`), each only in its form, else None."""
    reply = _object(body)
    code = _error(reply).get("code")
    collieai = reply.get("collieai")
    refusal = collieai.get("refusal") if isinstance(collieai, Mapping) else None
    rule_id = refusal.get("rule_id") if isinstance(refusal, Mapping) else None
    return (
        code if isinstance(code, str) and _CODE.fullmatch(code) else None,
        rule_id if isinstance(rule_id, str) and _RULE_ID.fullmatch(rule_id) else None,
    )


def _object(body: bytes) -> Mapping[str, object]:
    try:
        reply = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return {}
    return reply if isinstance(reply, dict) else {}


def _error(reply: Mapping[str, object]) -> Mapping[str, object]:
    error = reply.get("error")
    return error if isinstance(error, dict) else {}


def _findings(items: object) -> list[Mapping[str, object]]:
    """Every finding of the reply's items, as the diagnostics keep them: what is not an object is passed over."""
    found: list[Mapping[str, object]] = []
    for item in items if isinstance(items, list) else []:
        findings = item.get("findings") if isinstance(item, Mapping) else None
        found.extend(f for f in (findings if isinstance(findings, list) else []) if isinstance(f, Mapping))
    return found


def _category(rule_type: object) -> str:
    return rule_type if isinstance(rule_type, str) and rule_type in CATEGORIES else OTHER_CATEGORY


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
        kept: dict[str, object] = {
            "state": coverage.get("state"),
            "items_judged": coverage.get("items_judged"),
            "items_total": coverage.get("items_total"),
            "rules_run": _ints(coverage.get("rules_run")),
            "degraded_rules": _strs(coverage.get("degraded_rules")),
        }
        if "not_applicable" in coverage:  # a server of profile v2 lists the rules that did not apply, and why
            entries = coverage.get("not_applicable")
            kept["not_applicable"] = [
                {
                    "item": entry.get("item") if _is_int(entry.get("item")) else None,
                    "rule_id": _str_or_none(entry.get("rule_id")),
                    "reason": _str_or_none(entry.get("reason")),
                }
                for entry in (entries if isinstance(entries, list) else [])
                if isinstance(entry, Mapping)
            ]
        out["coverage"] = kept
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
            {
                **{key: _str_or_none(finding.get(key)) for key in ("rule_id", "rule_type", "action", "mode", "kind")},
                # An enforce reply says why a finding did not act — a mask is observed (`surface_mask_unsupported`).
                **(
                    {"no_enforce_reason": _str_or_none(finding.get("no_enforce_reason"))}
                    if "no_enforce_reason" in finding
                    else {}
                ),
            }
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
