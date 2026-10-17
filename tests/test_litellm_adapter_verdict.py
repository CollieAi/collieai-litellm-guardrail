"""The strict parse of a Monitor reply and the bypass reason of an error."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from typing import Any

import pytest

from collieai_guardrail.verdict import BYPASS_REASONS, Check, Rejected, error_reason, parse_check
from litellm_contract import ERROR_TABLE, check_reply


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
def test_a_check_is_flagged_when_enforce_would_have_masked_or_blocked(would: dict[str, Any], flagged: bool) -> None:
    reply = check_reply("pre_call")
    reply["collieai"]["would"].update(would)
    parsed = parse_check(_body(reply), phase="pre_call")
    assert isinstance(parsed, Check) and parsed.flagged is flagged


def test_a_check_without_its_would_is_still_a_check_and_not_flagged() -> None:
    reply = check_reply("pre_call")
    del reply["collieai"]["would"]
    parsed = parse_check(_body(reply), phase="pre_call")
    assert isinstance(parsed, Check) and parsed.flagged is False and parsed.would_outcome is None


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
