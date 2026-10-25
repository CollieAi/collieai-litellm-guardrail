"""The strict parse of a Monitor reply and the bypass reason of an error."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from typing import Any

import pytest

from collieai_guardrail.verdict import (
    BYPASS_REASONS,
    CATEGORIES,
    OTHER_CATEGORY,
    Check,
    Rejected,
    error_detail,
    error_reason,
    parse_check,
)
from litellm_contract import ERROR_TABLE, SECRET_TEXT, check_reply, legacy_reply


def _body(reply: Any) -> bytes:
    return json.dumps(reply).encode()


def test_the_contracts_example_is_a_check_with_its_diagnostics() -> None:
    parsed = parse_check(_body(check_reply("pre_call")), phase="pre_call")
    assert isinstance(parsed, Check)
    assert parsed.evaluation_id == "9f1c2d3e-0000-4000-8000-000000000001"
    assert parsed.coverage_state == "clean"
    assert parsed.would_outcome == "transform"
    assert parsed.diagnostics == check_reply("pre_call")["collieai"]


def test_a_degraded_coverage_is_a_check_too() -> None:
    reply = check_reply("post_call", items=1)
    reply["collieai"]["coverage"].update(state="degraded", degraded_rules=["r-7"])
    parsed = parse_check(_body(reply), phase="post_call")
    assert isinstance(parsed, Check) and parsed.coverage_state == "degraded"


def _set(path: str, value: Any) -> Callable[[dict[str, Any]], None]:
    def change(reply: dict[str, Any]) -> None:
        *parents, last = path.split(".")
        node = reply
        for key in parents:
            node = node[key]
        node[last] = value

    return change


def _drop(path: str) -> Callable[[dict[str, Any]], None]:
    def change(reply: dict[str, Any]) -> None:
        *parents, last = path.split(".")
        node = reply
        for key in parents:
            node = node[key]
        del node[last]

    return change


@pytest.mark.parametrize(
    ("change", "detail"),
    [
        (_drop("action"), "action"),  # stock generic reads a missing action as NONE; the adapter does not
        (_set("action", "none"), "action"),
        (_set("action", "BLOCKED"), "action"),
        (_set("action", "GUARDRAIL_INTERVENED"), "action"),
        (_set("texts", None), "replacement_key"),
        (_set("texts", ["masked"]), "replacement_key"),
        (_set("images", None), "replacement_key"),
        (_set("structured_messages", []), "replacement_key"),
        (_set("tools", None), "replacement_key"),
        (_set("tool_calls", None), "replacement_key"),
        (_set("stream_holdback_chars", 0), "replacement_key"),
        (_set("blocked_reason", None), "replacement_key"),
        (_drop("collieai"), "diagnostics_missing"),
        (_set("collieai", []), "diagnostics_missing"),
        (_set("collieai.version", 2), "version"),
        (_set("collieai.version", True), "version"),
        (_set("collieai.version", "1"), "version"),
        (_set("collieai.mode", "enforce"), "mode"),
        (_drop("collieai.mode"), "mode"),
        (_set("collieai.phase", "post_call"), "phase"),
        (_set("collieai.terminal", "deadline_expired"), "terminal"),
        (_set("collieai.terminal", "blocked"), "terminal"),
        (_set("collieai.coverage.state", "incomplete"), "coverage_state"),
        (_drop("collieai.coverage"), "coverage_state"),
        (_set("collieai.coverage.items_judged", 2), "items"),
        (_set("collieai.coverage.items_total", 0), "items"),
        (_set("collieai.coverage", {"state": "clean", "items_judged": 0, "items_total": 0, "rules_run": []}), "items"),
        (_set("collieai.coverage.items_judged", 3.0), "items"),
        (_set("collieai.coverage.items_judged", True), "items"),
        (_set("collieai.coverage.rules_run", [5, 5]), "rules_run"),
        (_set("collieai.coverage.rules_run", [5, 5, 5, 5]), "rules_run"),
        (_set("collieai.coverage.rules_run", None), "rules_run"),
        (_set("collieai.coverage.rules_run", [None, "oops", True]), "rules_run"),  # «3 judged» with no rules known
        (_set("collieai.coverage.rules_run", [5, -1, 5]), "rules_run"),
        (_set("collieai.coverage.rules_run", [5, 5.0, 5]), "rules_run"),
        (_set("collieai.coverage.rules_run", [5, False, 5]), "rules_run"),
        (_set("collieai.evaluation_id", ""), "evaluation_id"),
        (_set("collieai.evaluation_id", 7), "evaluation_id"),
        (_drop("collieai.evaluation_id"), "evaluation_id"),
    ],
)
def test_each_condition_of_the_strict_parse_failing_alone_is_not_a_check(
    change: Callable[[dict[str, Any]], None], detail: str
) -> None:
    reply = copy.deepcopy(check_reply("pre_call"))
    change(reply)
    assert parse_check(_body(reply), phase="pre_call") == Rejected(detail)


@pytest.mark.parametrize(
    ("body", "detail"),
    [
        (b"", "not_json"),
        (b"<html>ok</html>", "not_json"),
        (b"\xff\xfe", "not_json"),
        (b"[]", "not_object"),
        (b'"NONE"', "not_object"),
        (b"null", "not_object"),
    ],
)
def test_a_200_that_is_not_a_json_object_is_not_a_check(body: bytes, detail: str) -> None:
    assert parse_check(body, phase="pre_call") == Rejected(detail)


def test_the_diagnostics_keep_only_what_the_contract_defines() -> None:
    reply = check_reply("pre_call")
    collieai = reply["collieai"]
    collieai["text"] = "leaked text"
    collieai["refusal"] = {"path": "x"}
    collieai["profile"] = "p" * 300
    collieai["coverage"]["extra"] = "leaked"
    collieai["would"]["reason"] = "leaked"
    collieai["items"][0]["text"] = "leaked"
    collieai["items"][0]["findings"][0]["match"] = "leaked"
    collieai["items"].append("not an item")
    collieai["timings_ms"]["leaked text"] = "x"
    collieai["timings_ms"]["leaked key"] = 1.0  # a number under a name that is not a stage: the name is the leak
    collieai["timings_ms"]["judge"] = "leaked"
    parsed = parse_check(_body(reply), phase="pre_call")
    assert isinstance(parsed, Check)
    assert "leaked" not in json.dumps(parsed.diagnostics)
    assert "profile" not in parsed.diagnostics
    assert "refusal" not in parsed.diagnostics
    assert parsed.diagnostics["items"] == check_reply("pre_call")["collieai"]["items"]
    assert parsed.diagnostics["timings_ms"] == {"normalize": 0.4, "pin": 6.1, "check": 0.2, "total": 48.3}


@pytest.mark.parametrize(
    ("would", "flagged"),
    [
        ({}, True),  # the example: transform, a masked item
        ({"outcome": "allow", "masked_items": [], "would_mask": False}, False),
        ({"outcome": "none", "masked_items": [], "would_mask": False}, False),
        ({"outcome": "block", "masked_items": [], "would_mask": False, "would_block": True}, True),
        ({"outcome": "block", "masked_items": [], "would_mask": False}, True),
        ({"outcome": "allow", "masked_items": [], "would_mask": False, "would_block": True}, True),
        ({"outcome": "allow", "would_mask": True}, True),
        ({"outcome": "transform", "masked_items": [], "would_mask": False}, True),
    ],
)
def test_from_a_server_before_kinds_a_check_is_flagged_when_enforce_would_have_masked_or_blocked(
    would: dict[str, Any], flagged: bool
) -> None:
    reply = legacy_reply("pre_call")
    reply["collieai"]["would"].update(would)
    parsed = parse_check(_body(reply), phase="pre_call")
    assert isinstance(parsed, Check) and parsed.flagged is flagged
    assert parsed.categories == ()  # such a server does not say which finding is a detection


def test_a_check_without_its_would_is_still_a_check_and_from_a_server_before_kinds_not_flagged() -> None:
    reply = legacy_reply("pre_call")
    del reply["collieai"]["would"]
    parsed = parse_check(_body(reply), phase="pre_call")
    assert isinstance(parsed, Check) and parsed.flagged is False and parsed.would_outcome is None


# ------------------------------------------------------------------ the finding's kind


def _with_findings(*findings: dict[str, Any], would: str = "none") -> bytes:
    """The example with these findings on its one item, and what Enforce would have done."""
    reply = check_reply("pre_call")
    reply["collieai"]["items"][0]["findings"] = list(findings)
    reply["collieai"]["would"].update(outcome=would, masked_items=[], would_mask=False)
    return _body(reply)


def _finding(kind: Any = "detection", rule_type: Any = "aho_corasick", **extra: Any) -> dict[str, Any]:
    finding = {"rule_id": "r-1", "rule_type": rule_type, "action": "mask", "mode": "monitor", "kind": kind, **extra}
    if kind is ...:
        del finding["kind"]
    return finding


@pytest.mark.parametrize(
    ("findings", "would", "flagged", "categories"),
    [
        # a detection in Monitor: Enforce of this policy would have done nothing, the check still found it
        ([_finding()], "none", True, ("aho_corasick",)),
        ([_finding("allow")], "none", False, ()),
        ([_finding("operational_failure", "llm_detection")], "block", False, ()),  # fail-closed: no violation
        (
            [_finding("allow", "regex"), _finding(rule_type="structured_id"), _finding("operational_failure", "regex")],
            "transform",
            True,
            ("structured_id",),
        ),
        (
            [_finding(rule_type="regex"), _finding(rule_type="aho_corasick"), _finding(rule_type="regex")],
            "none",
            True,
            ("aho_corasick", "regex"),
        ),
    ],
)
def test_a_check_is_flagged_by_its_detections_and_names_their_rule_types(
    findings: list[dict[str, Any]], would: str, flagged: bool, categories: tuple[str, ...]
) -> None:
    parsed = parse_check(_with_findings(*findings, would=would), phase="pre_call")
    assert isinstance(parsed, Check)
    assert (parsed.flagged, parsed.categories) == (flagged, categories)


def test_a_check_with_no_finding_is_not_flagged_whatever_its_would_says() -> None:
    reply = check_reply("pre_call", found=False)
    reply["collieai"]["would"].update(outcome="block", would_block=True)
    parsed = parse_check(_body(reply), phase="pre_call")
    assert isinstance(parsed, Check) and (parsed.flagged, parsed.categories) == (False, ())


@pytest.mark.parametrize(
    "findings",
    [
        [_finding(), _finding(...)],  # some named, some not
        [_finding(...), _finding("allow")],
        [_finding("future_kind")],  # a word this adapter does not know
        [_finding("Detection")],
        [_finding(None)],
        [_finding(1)],
        [_finding(["detection"])],
        [_finding({"detection": True})],
    ],
)
def test_a_kind_named_on_some_findings_only_or_not_known_is_not_a_check(findings: list[dict[str, Any]]) -> None:
    assert parse_check(_with_findings(*findings), phase="pre_call") == Rejected("finding_kind")


def test_the_kinds_are_read_across_every_item() -> None:
    reply = check_reply("pre_call")
    reply["collieai"]["items"].append({"index": 0, "source": "x", "outcome": None, "findings": [_finding(...)]})
    assert parse_check(_body(reply), phase="pre_call") == Rejected("finding_kind")


@pytest.mark.parametrize(
    ("rule_type", "category"),
    [
        ("url_reputation", "url_reputation"),
        ("tool_access", OTHER_CATEGORY),  # a gate of tool calls, not a check of a message
        ("a_type_of_a_newer_server", OTHER_CATEGORY),
        (SECRET_TEXT, OTHER_CATEGORY),
        ("", OTHER_CATEGORY),
        (None, OTHER_CATEGORY),
        (7, OTHER_CATEGORY),
        (["regex"], OTHER_CATEGORY),
    ],
)
def test_a_category_is_a_known_rule_type_or_other_never_the_servers_word(rule_type: Any, category: str) -> None:
    parsed = parse_check(_with_findings(_finding(rule_type=rule_type)), phase="pre_call")
    assert isinstance(parsed, Check) and parsed.categories == (category,)


def test_the_categories_are_bounded_by_the_rule_types_and_other() -> None:
    every = [_finding(rule_type=rule_type) for rule_type in [*sorted(CATEGORIES), "new_a", "new_b", "tool_access"]]
    parsed = parse_check(_with_findings(*every), phase="pre_call")
    assert isinstance(parsed, Check)
    assert parsed.categories == tuple(sorted({*CATEGORIES, OTHER_CATEGORY}))
    assert len(parsed.categories) == 12 and all(len(category) <= 32 for category in parsed.categories)


def test_the_diagnostics_keep_each_findings_kind_and_the_rules_that_did_not_apply() -> None:
    reply = check_reply("pre_call")
    reply["collieai"]["coverage"]["not_applicable"] = [
        {"item": 0, "rule_id": "r1", "reason": "no_supported_urls", "text": "leaked"},
        {"item": "0", "rule_id": 5, "reason": "r" * 300},
        "leaked",
    ]
    parsed = parse_check(_body(reply), phase="pre_call")
    assert isinstance(parsed, Check)
    assert parsed.diagnostics["items"] == check_reply("pre_call")["collieai"]["items"]  # `kind` kept
    assert parsed.diagnostics["coverage"]["not_applicable"] == [  # type: ignore[index]
        {"item": 0, "rule_id": "r1", "reason": "no_supported_urls"},
        {"item": None, "rule_id": None, "reason": None},
    ]
    assert "leaked" not in json.dumps(parsed.diagnostics)
    del reply["collieai"]["coverage"]["not_applicable"]  # profile v1: the server does not say
    parsed = parse_check(_body(reply), phase="pre_call")
    assert isinstance(parsed, Check) and "not_applicable" not in parsed.diagnostics["coverage"]  # type: ignore[operator]


@pytest.mark.parametrize(("status", "body", "reason"), ERROR_TABLE)
def test_each_row_of_the_error_table_has_its_bypass_reason(status: int, body: Any, reason: str) -> None:
    assert reason in BYPASS_REASONS
    assert error_reason(status, _body(body)) == reason


@pytest.mark.parametrize(
    ("status", "body", "reason"),
    [
        (502, b"<html>Bad Gateway</html>", "http_5xx"),  # a load balancer, not the server
        (503, b"", "http_5xx"),
        (504, b"upstream timed out", "http_5xx"),
        (400, b"{}", "http_4xx"),
        (422, _body({"detail": [{"loc": ["body"]}]}), "http_4xx"),  # no error.type
        (422, _body({"error": {"type": 5}}), "http_4xx"),
        (418, _body({"error": {"type": "teapot"}}), "http_4xx"),
        (503, _body({"error": {"type": "deadline"}}), "http_5xx"),  # a type the table does not pair with 503
        (500, _body({"error": {"type": "busy"}}), "http_5xx"),
        (302, b"", "http_3xx"),
        (429, b"slow down", "rate_limited"),
    ],
)
def test_a_reply_outside_the_table_is_named_by_its_status_class(status: int, body: bytes, reason: str) -> None:
    assert error_reason(status, body) == reason


@pytest.mark.parametrize(("status", "body", "reason"), ERROR_TABLE)
def test_an_errors_detail_is_its_code(status: int, body: Any, reason: str) -> None:
    assert error_detail(_body(body)) == (body["error"].get("code"), None)


def test_a_refusals_detail_names_its_rule() -> None:
    body = {
        "error": {
            "message": "Policy is outside the supported execution profile",
            "type": "capability",
            "code": "aho_source",
        },
        "collieai": {
            "version": 1,
            "terminal": "refused",
            "refusal": {"kind": "capability", "reason": "aho_source", "rule_id": "6f1c-rule.2:a_b"},
        },
    }
    assert error_detail(_body(body)) == ("aho_source", "6f1c-rule.2:a_b")


@pytest.mark.parametrize(
    ("code", "rule_id"),
    [
        ("Aho source", "has space"),
        (SECRET_TEXT, SECRET_TEXT),
        ("a" * 65, "r" * 129),
        ("", ""),
        ("_lead", "-lead"),
        ("code\n", "rule\nid"),
        (5, 5),
        (None, None),
    ],
)
def test_a_detail_outside_its_form_is_left_out(code: Any, rule_id: Any) -> None:
    body = {"error": {"type": "capability", "code": code}, "collieai": {"refusal": {"rule_id": rule_id}}}
    assert error_detail(_body(body)) == (None, None)


@pytest.mark.parametrize("body", [b"", b"<html>Bad Gateway</html>", b"[]", b'{"error": "x", "collieai": []}'])
def test_a_reply_without_its_error_object_has_no_detail(body: bytes) -> None:
    assert error_detail(body) == (None, None)


def test_a_reply_with_rules_that_did_not_apply_is_still_a_check() -> None:
    """A newer server lists the rules that did not apply to the text (`coverage.not_applicable`); this
    adapter reads past the field, and the check is as before."""
    reply = check_reply("pre_call")
    reply["collieai"]["coverage"]["not_applicable"] = [{"item": 0, "rule_id": "r1", "reason": "no_supported_urls"}]

    parsed = parse_check(_body(reply), phase="pre_call")

    assert isinstance(parsed, Check) and parsed.coverage_state == "clean"
