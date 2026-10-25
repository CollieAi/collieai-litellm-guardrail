"""What the adapter reads from LiteLLM's data, its count apart from the hooks, the summary and the synthetic check
None of it imports LiteLLM, so it runs anywhere;
the same functions against LiteLLM's real objects are in `test_litellm_adapter_glue.py`."""

from __future__ import annotations

import ast
import asyncio
import enum
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import collieai_guardrail.heartbeat as heartbeat_module
from collieai_guardrail.config import AdapterConfig, ConfigError, guardrail_name
from collieai_guardrail.counters import Counters
from collieai_guardrail.heartbeat import (
    SYNTHETIC_BUSY_RETRIES,
    SYNTHETIC_START_SPREAD_S,
    SYNTHETIC_TEXT,
    Heartbeat,
    interval_counts,
    only_busy,
    synthetic_check,
    unwired_phases,
)
from collieai_guardrail.litellm_data import (
    OPERATION_ID_KEY,
    OPERATION_TAG_KEY,
    LoggedCalls,
    call_facts,
    call_type_of,
    conversation_id,
    count_logged,
    issue_operation_id,
    logged_operation_id,
    logged_request,
)
from collieai_guardrail.observer import Observer, Outcome
from collieai_guardrail.payload import CallFacts
from litellm_contract import (
    API_BASE,
    API_KEY,
    SECRET_TEXT,
    MockServer,
    chat_request,
    chat_response,
    check_reply,
    json_reply,
    refuses_connection,
)


class _CallTypes(str, enum.Enum):  # noqa: UP042 — the shape of LiteLLM's `CallTypes`, not a StrEnum
    acompletion = "acompletion"


# ------------------------------------------------------------------ LiteLLM's data


def test_the_call_type_is_read_from_an_enum_member_or_a_string() -> None:
    assert call_type_of(_CallTypes.acompletion) == "acompletion"
    assert call_type_of("aresponses") == "aresponses"
    assert call_type_of(None) is None
    assert call_type_of("") is None
    assert call_type_of(5) is None


def test_the_facts_come_from_the_request_and_the_logging_object() -> None:
    logging_obj = SimpleNamespace(litellm_call_id="call-from-logging", litellm_trace_id="trace-1")
    data = chat_request(
        litellm_logging_obj=logging_obj,
        proxy_server_request={"headers": {"X-CollieAI-Conversation-ID": "conv-9", "authorization": "Bearer x"}},
    )
    facts = call_facts(data, operation_id="op-1")
    assert facts.operation_id == "op-1"
    assert facts.litellm_call_id == "call-1"  # the request's own
    assert facts.litellm_trace_id == "trace-1"
    assert facts.model == "gpt-4o-mini"
    assert facts.conversation_id == "conv-9"
    del data["litellm_call_id"]
    assert call_facts(data, operation_id="op-1").litellm_call_id == "call-from-logging"


@pytest.mark.parametrize(
    "data",
    [
        {},
        {"litellm_logging_obj": object(), "proxy_server_request": None},
        {"litellm_call_id": 5, "model": ["x"], "proxy_server_request": {"headers": "x"}},
        {"litellm_call_id": "c" * 300, "proxy_server_request": {"headers": {"x-collieai-conversation-id": 7}}},
    ],
)
def test_facts_of_an_unexpected_shape_are_absent_not_an_error(data: dict[str, Any]) -> None:
    facts = call_facts(data, operation_id="op-1")
    assert (facts.litellm_call_id, facts.litellm_trace_id, facts.model, facts.conversation_id) == (None,) * 4


def test_the_conversation_id_is_only_that_header() -> None:
    assert conversation_id({"proxy_server_request": {"headers": {"x-other": "v"}}}) is None
    assert conversation_id({"proxy_server_request": {"headers": {"x-collieai-conversation-id": "c-1"}}}) == "c-1"


def test_a_logged_call_gives_the_fields_eligibility_reads() -> None:
    kwargs = {
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
        "optional_params": {"n": 2, "temperature": 0.1, "tools": [{"type": "function"}]},
        "litellm_params": {"metadata": {"user_api_key_alias": "a"}},
    }
    assert logged_request(kwargs) == {
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
        "n": 2,
        "tools": [{"type": "function"}],
    }


def _logged(**extra: Any) -> dict[str, Any]:
    request = chat_request()
    return {
        "call_type": "acompletion",
        "messages": request["messages"],
        "stream": False,
        "optional_params": {},
        **extra,
    }


def test_logged_calls_are_counted_per_phase_apart_from_the_hooks() -> None:
    counters = Counters()
    count_logged(counters, _logged(), chat_response(), succeeded=True)
    count_logged(counters, _logged(stream=True), None, succeeded=True)
    count_logged(counters, _logged(call_type=_CallTypes.acompletion), None, succeeded=False)  # no post-call phase
    count_logged(counters, _logged(call_type="aembedding"), {"object": "list"}, succeeded=True)
    count_logged(counters, _logged(), chat_response("safe", reasoning_content="x"), succeeded=True)
    pre, post = counters.snapshot()["pre_call"], counters.snapshot()["post_call"]
    assert (pre["logged_calls"], pre["logged_eligible"]) == (5, 3)
    assert pre["logged_outside"] == {"call_type": 1, "streaming": 1}
    assert (post["logged_calls"], post["logged_eligible"]) == (4, 1)
    assert post["logged_outside"] == {"call_type": 1, "reasoning": 1, "streaming": 1}
    assert pre["hook_calls"] == pre["eligible"] == 0  # the hooks' counts are their own


def test_a_call_the_proxy_refused_before_any_model_is_logged_outside_as_rejected() -> None:
    """LiteLLM logs a refusal (a bad key, a limit, a budget, a guardrail's block) as a failed call with its messages,
    marked `litellm_no_upstream_llm_call`; its hooks need not have run."""
    counters = Counters()
    count_logged(counters, _logged(litellm_no_upstream_llm_call=True), None, succeeded=False)
    count_logged(counters, _logged(litellm_no_upstream_llm_call=False), None, succeeded=False)
    pre, post = counters.snapshot()["pre_call"], counters.snapshot()["post_call"]
    assert (pre["logged_calls"], pre["logged_eligible"], pre["logged_outside"]) == (2, 1, {"rejected": 1})
    assert post["logged_calls"] == 0


def _issued() -> dict[str, object]:
    """A request's metadata once pre-call ran: the ID it issued, with its tag."""
    metadata: dict[str, object] = {"user_api_key_alias": "a"}
    issue_operation_id(metadata)
    return metadata


def _of(metadata: object, **extra: Any) -> dict[str, Any]:
    """A log event of the request with this metadata, where LiteLLM keeps it."""
    return _logged(litellm_params={"metadata": metadata}, **extra)


def test_a_call_the_router_retried_is_counted_once_with_its_repeats() -> None:
    """The router logs a failed attempt, then the retry's outcome, each with the request's metadata and in it the ID
    pre-call issued (seen on the real proxy)."""
    a, b, c = _issued(), _issued(), _issued()
    counters, calls = Counters(), LoggedCalls()
    count_logged(counters, _of(a), None, succeeded=False, calls=calls)
    count_logged(counters, _of(a), chat_response(), succeeded=True, calls=calls)
    count_logged(counters, _of(a), chat_response(), succeeded=True, calls=calls)
    count_logged(counters, _of(b), None, succeeded=False, calls=calls)
    count_logged(counters, _of(b), None, succeeded=False, calls=calls)
    for succeeded in (True, False, True):  # a failure after a success does not make the next success new
        count_logged(counters, _of(c), chat_response(), succeeded=succeeded, calls=calls)
    count_logged(counters, _logged(), chat_response(), succeeded=True, calls=calls)  # no ID: every event counts
    count_logged(counters, _logged(), chat_response(), succeeded=True, calls=calls)
    pre, post = counters.snapshot()["pre_call"], counters.snapshot()["post_call"]
    assert (pre["logged_calls"], pre["logged_eligible"], pre["logged_repeats"]) == (5, 5, 5)
    assert (post["logged_calls"], post["logged_eligible"]) == (4, 4)
    assert post["logged_repeats"] == 0


def test_calls_under_one_caller_chosen_call_id_are_counted_apart() -> None:
    """The proxy takes `litellm_call_id` from the caller's `x-litellm-call-id` header, and a client may send one
    for a whole session: it joins no events."""
    counters, calls = Counters(), LoggedCalls()
    for metadata in (_issued(), _issued(), "no metadata", None):
        count_logged(counters, _of(metadata, litellm_call_id="session-1"), chat_response(), succeeded=True, calls=calls)
    pre, post = counters.snapshot()["pre_call"], counters.snapshot()["post_call"]
    assert (pre["logged_calls"], pre["logged_repeats"], post["logged_calls"]) == (4, 0, 4)


_ANOTHER = _issued()
CARD_IN_HEX = b"4111111111111111".hex()  # a card number, written by the caller in the ID's very form


@pytest.mark.parametrize(
    "claimed",
    [
        {OPERATION_ID_KEY: CARD_IN_HEX},
        {OPERATION_ID_KEY: CARD_IN_HEX, OPERATION_TAG_KEY: "0" * 32},
        {OPERATION_ID_KEY: CARD_IN_HEX, OPERATION_TAG_KEY: _ANOTHER[OPERATION_TAG_KEY]},  # another ID's tag
        {OPERATION_ID_KEY: CARD_IN_HEX, OPERATION_TAG_KEY: "PELICAN é"},
        {OPERATION_ID_KEY: 7, OPERATION_TAG_KEY: 7},
        {OPERATION_ID_KEY: "PELICAN é", OPERATION_TAG_KEY: "0" * 32},  # not the ID's form: no tag is even made
        {OPERATION_ID_KEY: str(_ANOTHER[OPERATION_ID_KEY]).upper(), OPERATION_TAG_KEY: _ANOTHER[OPERATION_TAG_KEY]},
    ],
)
def test_an_id_without_pre_calls_tag_joins_no_events(claimed: dict[str, object]) -> None:
    counters, calls = Counters(), LoggedCalls()
    for _ in range(2):
        count_logged(counters, _of(dict(claimed)), None, succeeded=False, calls=calls)
    pre = counters.snapshot()["pre_call"]
    assert (pre["logged_calls"], pre["logged_repeats"]) == (2, 0)


def test_pre_calls_id_is_read_where_litellm_logs_the_metadata() -> None:
    issued = _issued()
    operation_id = issued[OPERATION_ID_KEY]
    assert logged_operation_id({"litellm_params": {"metadata": issued}}) == operation_id
    assert logged_operation_id({"litellm_params": {"metadata": {}, "litellm_metadata": issued}}) == operation_id
    assert logged_operation_id({"litellm_params": {"metadata": {"x": 1}, "litellm_metadata": issued}}) == operation_id
    assert logged_operation_id({"litellm_params": {"metadata": issued, "litellm_metadata": {"x": 1}}}) == operation_id
    assert logged_operation_id({"metadata": issued}) is None  # not where LiteLLM logs it
    assert logged_operation_id({"litellm_params": "x"}) is None
    assert logged_operation_id({}) is None


def test_the_calls_kept_are_bounded_oldest_first_out() -> None:
    a, b, c = _issued(), _issued(), _issued()
    counters, calls = Counters(), LoggedCalls(capacity=2)
    for metadata in (a, b, a, c, b):
        count_logged(counters, _of(metadata), None, succeeded=False, calls=calls)
    pre = counters.snapshot()["pre_call"]
    assert (pre["logged_calls"], pre["logged_repeats"]) == (4, 1)  # b had gone when it came back


def test_counting_a_logged_call_of_any_shape_never_raises() -> None:
    counters = Counters()
    count_logged(counters, {"messages": object(), "optional_params": "x"}, object(), succeeded=True)
    count_logged(counters, {}, None, succeeded=False)
    assert counters.snapshot()["pre_call"]["logged_outside"] == {"call_type": 2}


# ------------------------------------------------------------------ settings


@pytest.mark.parametrize(
    "change",
    [
        {"summary_interval_s": 0},
        {"summary_interval_s": 3601},
        {"synthetic_interval_s": -1},
        {"synthetic_interval_s": True},
    ],
)
def test_the_heartbeat_settings_are_checked(change: dict[str, Any]) -> None:
    with pytest.raises(ConfigError):
        AdapterConfig(api_base=API_BASE, api_key=API_KEY, **change)


@pytest.mark.parametrize("api_key", ["sk-0123456789abcdef", "clai_"])
def test_a_key_that_is_not_a_collieai_project_key_is_refused(api_key: str) -> None:
    with pytest.raises(ConfigError, match="api_key") as refused:
        AdapterConfig(api_base=API_BASE, api_key=api_key)
    assert api_key == "clai_" or api_key not in str(refused.value)  # the message names the prefix itself


@pytest.mark.parametrize("api_key", [None, "", "None"])
def test_an_unset_key_names_the_variable_to_set(api_key: Any) -> None:
    """LiteLLM resolves `os.environ/COLLIEAI_API_KEY` of an unset variable to None (the proxy's start, LiteLLM
    1.102.4 image), "" or the string "None" (its registry); each says which variable to set."""
    with pytest.raises(ConfigError, match=r"api_key is not set.*COLLIEAI_API_KEY"):
        AdapterConfig(api_base=API_BASE, api_key=api_key)


@pytest.mark.parametrize("api_key", [" clai_0123456789", "clai_0123456789 "])
def test_a_key_with_surrounding_whitespace_is_refused_without_its_value(api_key: str) -> None:
    with pytest.raises(ConfigError, match="surrounding whitespace") as refused:
        AdapterConfig(api_base=API_BASE, api_key=api_key)
    assert api_key.strip() not in str(refused.value)


def test_a_setting_error_is_not_one_litellms_loader_swallows() -> None:
    """`init_guardrails_v2` catches ValueError and TypeError and starts the proxy without the guardrail."""
    assert not issubclass(ConfigError, (ValueError, TypeError))


# ------------------------------------------------------------------ the summary


def _snapshot(**pre: Any) -> dict[str, dict[str, object]]:
    counters = Counters()
    for name, n in pre.items():
        for _ in range(n):
            if name.startswith("outside_"):
                counters.add_reason("pre_call", "outside", name.removeprefix("outside_"))
            elif name.startswith("bypass_"):
                counters.add_reason("pre_call", "bypass", name.removeprefix("bypass_"))
            else:
                counters.add("pre_call", name)
    return counters.snapshot()


def test_the_summary_counts_the_interval_with_coverage_and_the_outside_share() -> None:
    before = _snapshot(hook_calls=4, eligible=3, completed=3, logged_eligible=3, outside_streaming=1)
    after = _snapshot(
        hook_calls=14,
        eligible=9,
        attempted=6,
        completed=6,
        bypass_busy=2,
        logged_eligible=10,
        outside_streaming=3,
        outside_reasoning=2,
    )
    pre = interval_counts(before, after)["pre_call"]
    assert (pre["hook_calls"], pre["eligible"], pre["completed"]) == (10, 6, 3)
    assert pre["bypass"] == {"busy": 2}
    assert pre["outside"] == {"reasoning": 2, "streaming": 2}
    assert pre["coverage"] == 0.6  # 3 of the 5 checks that ended
    assert pre["outside_share"] == 0.4  # 4 of the 10 calls the hook saw were not observed
    assert pre["logged_unseen"] == 1
    empty = interval_counts(after, after)["post_call"]
    assert empty["coverage"] is None and empty["outside_share"] is None
    quiet = interval_counts(after, after)["pre_call"]
    assert quiet["bypass"] == {} and quiet["outside"] == {}  # a reason with nothing new in the interval is left out


def test_coverage_is_of_the_checks_that_ended_in_the_interval_not_of_those_that_started() -> None:
    crossing = _snapshot(hook_calls=1, eligible=1, attempted=1)  # a check still running at the interval's end
    ended = _snapshot(hook_calls=2, eligible=2, attempted=2, completed=2)  # it ends, and one more starts and ends
    assert interval_counts(crossing, ended)["pre_call"]["coverage"] == 1.0  # not 2.0: two ended, two completed
    only_ended = _snapshot(hook_calls=1, eligible=1, attempted=1, completed=1)  # no new start in the interval
    assert interval_counts(crossing, only_ended)["pre_call"]["coverage"] == 1.0  # not None
    failed = _snapshot(hook_calls=1, eligible=1, bypass_timeout=1)
    assert interval_counts(crossing, failed)["pre_call"]["coverage"] == 0.0


def test_a_phase_whose_hook_has_never_run_while_litellm_logged_eligible_calls_is_unwired() -> None:
    logged = _snapshot(logged_eligible=3)
    assert unwired_phases(interval_counts(Counters().snapshot(), logged), logged) == ["pre_call"]
    seen = _snapshot(logged_eligible=3, hook_calls=1)
    assert unwired_phases(interval_counts(Counters().snapshot(), seen), seen) == []
    assert unwired_phases(interval_counts(Counters().snapshot(), Counters().snapshot()), Counters().snapshot()) == []
    # the hook ran in the last interval, LiteLLM logs the call in this one: wired
    hook_ran, call_logged = _snapshot(hook_calls=1, eligible=1), _snapshot(hook_calls=1, eligible=1, logged_eligible=1)
    counts = interval_counts(hook_ran, call_logged)
    assert counts["pre_call"]["hook_calls"] == 0 and counts["pre_call"]["logged_eligible"] == 1
    assert unwired_phases(counts, call_logged) == []


def _observer(server: MockServer, counters: Counters | None = None) -> Observer:
    config = AdapterConfig(api_base=API_BASE, api_key=API_KEY, deadline_ms=1000, network_margin_ms=100)
    return Observer(config, litellm_version="1.102.4", client=server.client(), counters=counters)


async def test_the_summary_line_is_json_of_counts_only_and_warns_about_an_unwired_phase(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="collieai_guardrail")
    counters = Counters()
    heartbeat = Heartbeat(
        _observer(MockServer(json_reply(200, check_reply()))), counters, summary_interval_s=60, synthetic_interval_s=0
    )
    count_logged(counters, _logged(), chat_response(), succeeded=True)
    summary = heartbeat.summarize()
    assert summary is not None
    line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("collieai guardrail summary "))
    assert json.loads(line.removeprefix("collieai guardrail summary ")) == summary
    pre = summary["pre_call"]
    assert isinstance(pre, dict) and pre["logged_eligible"] == 1 and pre["logged_unseen"] == 1
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 2 and all("not wired" in w for w in warnings)
    text = "\n".join(r.getMessage() for r in caplog.records)
    for secret in (SECRET_TEXT, "PELICAN", API_KEY, "call-1"):
        assert secret not in text
    caplog.clear()
    second = heartbeat.summarize()  # the next interval starts from this one
    assert second is not None and second["pre_call"]["logged_eligible"] == 0  # type: ignore[index]
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]
    counters.add("pre_call", "hook_calls")  # the hook ran in this interval ...
    heartbeat.summarize()
    count_logged(counters, _logged(), None, succeeded=False)  # ... LiteLLM logs its call in the next
    caplog.clear()
    heartbeat.summarize()
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]


def test_each_guardrails_summary_names_it(caplog: pytest.LogCaptureFixture) -> None:
    """With two CollieAi guardrails in one proxy, each summary says whose counts it has."""
    caplog.set_level(logging.INFO, logger="collieai_guardrail")
    for name in ("collieai-team-a", "collieai-team-b", None):
        observer = _observer(MockServer(json_reply(200, check_reply())))
        Heartbeat(observer, Counters(), summary_interval_s=60, synthetic_interval_s=0, name=name).summarize()
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("collieai guardrail summary ")]
    names = [json.loads(line.removeprefix("collieai guardrail summary "))["guardrail"] for line in lines]
    assert names == ["collieai-team-a", "collieai-team-b", None]


def test_every_line_an_instance_logs_ends_with_its_guardrails_name() -> None:
    """Each WARNING and INFO of the glue, the heartbeat and the observer passes the name's tail as
    its last argument, the summary aside, whose JSON carries the name. A line added without it fails here."""
    package = Path(__file__).resolve().parent.parent / "collieai_guardrail"
    lines = 0
    for module in ("guardrail.py", "heartbeat.py", "observer.py"):
        for node in ast.walk(ast.parse((package / module).read_text())):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("warning", "info", "error")
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "logger"
            ):
                continue
            text = node.args[0]
            assert isinstance(text, ast.Constant) and isinstance(text.value, str)
            if text.value == "collieai guardrail summary %s":
                continue
            last = node.args[-1]
            lines += 1
            assert text.value.endswith("%s") and isinstance(last, ast.Attribute) and last.attr == "_named", (
                f"{module}:{node.lineno}"
            )
    assert lines == 14


@pytest.mark.parametrize(
    ("value", "name"),
    [
        ("collieai-monitor", "collieai-monitor"),
        ("CollieAi monitor · team A", "CollieAi monitor · team A"),
        ("n" * 128, "n" * 128),
        ("n" * 129, None),
        ("two\nlines", None),
        ("tab\there", None),
        ("", None),
        (None, None),
        (7, None),
    ],
)
def test_the_name_the_log_repeats_is_one_printable_line(value: Any, name: str | None) -> None:
    assert guardrail_name(value) == name


# ------------------------------------------------------------------ the synthetic check


async def test_the_synthetic_check_observes_both_phases_with_a_fixed_text_and_counts_nothing() -> None:
    replies = iter([check_reply("pre_call", items=1), check_reply("post_call", items=1)])
    server = MockServer(lambda request: json_reply(200, next(replies))(request))
    counters = Counters()
    pre, post = await asyncio.wait_for(synthetic_check(_observer(server, counters)), timeout=5)
    assert (pre.result, post.result) == ("completed", "completed")
    assert server.bodies[0]["structured_messages"] == [{"role": "user", "content": SYNTHETIC_TEXT}]
    assert server.bodies[1]["texts"] == [SYNTHETIC_TEXT]
    operation_ids = {b["additional_provider_specific_params"]["collieai"]["operation_id"] for b in server.bodies}
    assert len(operation_ids) == 1 and next(iter(operation_ids)).startswith("synthetic-")
    assert counters.snapshot() == Counters().snapshot()


async def test_a_failing_synthetic_check_warns_with_each_phases_reason(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="collieai_guardrail")
    heartbeat = Heartbeat(
        _observer(MockServer(json_reply(401, {"error": {"type": "authentication_error"}}))),
        Counters(),
        summary_interval_s=60,
        synthetic_interval_s=300,
    )
    await asyncio.wait_for(heartbeat.run_synthetic(), timeout=5)
    warning = next(
        r.getMessage() for r in caplog.records if r.levelno == logging.WARNING and "synthetic" in r.getMessage()
    )
    assert "pre_call bypass:auth" in warning and "post_call bypass:auth" in warning
    summary = heartbeat.summarize()
    assert summary is not None and summary["synthetic"] == {"pre_call": "bypass:auth", "post_call": "bypass:auth"}


async def test_a_synthetic_check_refused_for_a_rule_names_the_code_and_the_rule(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Before, the WARNING said `capability` and `detail=None`; which rule and why were only in CollieAi."""
    caplog.set_level(logging.INFO, logger="collieai_guardrail")
    refused = {
        "error": {
            "message": "Policy is outside the supported execution profile",
            "type": "capability",
            "code": "aho_source",
        },
        "collieai": {"version": 1, "terminal": "refused", "refusal": {"kind": "capability", "rule_id": "6f1c-rule"}},
    }
    heartbeat = Heartbeat(
        _observer(MockServer(json_reply(422, refused))),
        Counters(),
        summary_interval_s=60,
        synthetic_interval_s=300,
        name="collieai-monitor",
    )
    await asyncio.wait_for(heartbeat.run_synthetic(), timeout=5)
    [warning] = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert "pre_call bypass:capability (detail=aho_source, http_status=422, rule_id=6f1c-rule)" in warning
    assert "post_call bypass:capability (detail=aho_source, http_status=422, rule_id=6f1c-rule)" in warning
    assert warning.endswith("(guardrail collieai-monitor)")


async def test_a_synthetic_check_that_fails_only_post_call_warns_too(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="collieai_guardrail")
    replies = iter([json_reply(200, check_reply("pre_call", items=1)), json_reply(503, {"error": {}})])
    heartbeat = Heartbeat(
        _observer(MockServer(lambda request: next(replies)(request))),
        Counters(),
        summary_interval_s=60,
        synthetic_interval_s=300,
    )
    await asyncio.wait_for(heartbeat.run_synthetic(), timeout=5)
    [record] = [r for r in caplog.records if "synthetic check" in r.getMessage()]
    assert record.levelno == logging.WARNING
    assert "pre_call completed" in record.getMessage() and "post_call bypass:" in record.getMessage()


async def test_a_passing_synthetic_check_logs_at_info(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="collieai_guardrail")
    replies = iter([check_reply("pre_call", items=1), check_reply("post_call", items=1)])
    heartbeat = Heartbeat(
        _observer(MockServer(lambda request: json_reply(200, next(replies))(request))),
        Counters(),
        summary_interval_s=60,
        synthetic_interval_s=300,
        name="collieai-monitor",
    )
    await asyncio.wait_for(heartbeat.run_synthetic(), timeout=5)
    assert [r.getMessage() for r in caplog.records if r.name == "collieai_guardrail"] == [
        "collieai guardrail synthetic check: pre_call completed, post_call completed (guardrail collieai-monitor)"
    ]


# ------------------------------------------------------------------ the loop


async def test_the_loop_runs_the_synthetic_check_at_start_and_summarizes_on_time(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="collieai_guardrail")
    server = MockServer(refuses_connection)
    heartbeat = Heartbeat(
        _observer(server), Counters(), summary_interval_s=1, synthetic_interval_s=3600, start_spread_s=0
    )
    heartbeat.ensure_running()
    heartbeat.ensure_running()  # once per event loop
    try:
        for _ in range(300):
            if any(r.getMessage().startswith("collieai guardrail summary") for r in caplog.records):
                break
            await asyncio.sleep(0.01)
    finally:
        await heartbeat.stop()
    assert len(server.requests) == 2  # one synthetic check: pre and post, once
    assert any(r.getMessage().startswith("collieai guardrail summary") for r in caplog.records)
    heartbeat.ensure_running()  # stopped for good
    assert heartbeat._task is None


@pytest.mark.parametrize(("interval_s", "spread_s"), [(300, SYNTHETIC_START_SPREAD_S), (10, 1.0), (0, None)])
async def test_the_first_synthetic_check_waits_a_random_part_of_its_spread(
    monkeypatch: pytest.MonkeyPatch, interval_s: int, spread_s: float | None
) -> None:
    """Workers that start together would send theirs at one moment; a server with one check at a time per process
    answers some `busy` (seen with two proxy workers)."""
    drawn: list[tuple[float, float]] = []

    def uniform(low: float, high: float) -> float:
        drawn.append((low, high))
        return 3600.0  # the check does not come within the test

    monkeypatch.setattr(heartbeat_module.random, "uniform", uniform)
    server = MockServer(refuses_connection)
    heartbeat = Heartbeat(_observer(server), Counters(), summary_interval_s=3600, synthetic_interval_s=interval_s)
    heartbeat.ensure_running()
    await asyncio.sleep(0.05)
    await heartbeat.stop()
    assert drawn == ([] if spread_s is None else [(0, spread_s)])
    assert server.requests == []


#: The server's reply when it has no free slot.
BUSY = json_reply(503, {"error": {"message": "Guardrail evaluation capacity is busy", "type": "busy", "code": "busy"}})


def _pre_call_answers(*answers: str) -> MockServer:
    """Pre-call answered in turn `busy` or `ok`, and `ok` once they run out; post-call always completes."""
    planned = list(answers)

    def respond(request: Any) -> Any:
        phase = "pre_call" if json.loads(request.content)["input_type"] == "request" else "post_call"
        if phase == "pre_call" and planned and planned.pop(0) == "busy":
            return BUSY(request)
        return json_reply(200, check_reply(phase, items=1))(request)

    return MockServer(respond)


async def _until_warned(caplog: pytest.LogCaptureFixture, heartbeat: Heartbeat) -> None:
    heartbeat.ensure_running()
    try:
        for _ in range(500):
            if any(r.levelno == logging.WARNING for r in caplog.records):
                return
            await asyncio.sleep(0.01)
    finally:
        await heartbeat.stop()


async def test_a_synthetic_check_answered_busy_is_repeated_before_it_warns(caplog: pytest.LogCaptureFixture) -> None:
    """Workers that start together may still meet at a server that admits one check at a time: a
    `busy` is repeated; busy every time is the server's capacity and warns. A pass starts the retries afresh."""
    caplog.set_level(logging.INFO, logger="collieai_guardrail")
    server = _pre_call_answers("busy", "ok", "busy", "busy", "busy")
    heartbeat = Heartbeat(
        _observer(server), Counters(), summary_interval_s=3600, synthetic_interval_s=1, start_spread_s=0
    )
    await _until_warned(caplog, heartbeat)
    lines = [(r.levelname, r.getMessage().split(";")[0]) for r in caplog.records if "synthetic check" in r.getMessage()]
    busy = ("INFO", "collieai guardrail synthetic check: pre_call bypass:busy, post_call completed")
    passed = ("INFO", "collieai guardrail synthetic check: pre_call completed, post_call completed")  # no name
    assert lines[:-1] == [busy, passed, *[busy] * SYNTHETIC_BUSY_RETRIES]
    assert lines[-1][0] == "WARNING" and "pre_call bypass:busy" in lines[-1][1]
    assert len(server.requests) == 2 * len(lines)


async def test_a_busy_synthetic_check_is_repeated_after_a_random_wait_from_half_the_spread(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="collieai_guardrail")
    drawn: list[tuple[float, float]] = []

    def uniform(low: float, high: float) -> float:
        drawn.append((low, high))
        return 0.0

    monkeypatch.setattr(heartbeat_module.random, "uniform", uniform)
    server = _pre_call_answers(*["busy"] * (1 + SYNTHETIC_BUSY_RETRIES))
    heartbeat = Heartbeat(_observer(server), Counters(), summary_interval_s=3600, synthetic_interval_s=300)
    await _until_warned(caplog, heartbeat)
    spread = SYNTHETIC_START_SPREAD_S
    assert drawn == [(0, spread), *[(spread / 2, spread)] * SYNTHETIC_BUSY_RETRIES]
    assert len(server.requests) == 2 * (1 + SYNTHETIC_BUSY_RETRIES)  # the next check is an interval away


async def test_a_synthetic_checks_bypass_is_left_to_the_heartbeat(caplog: pytest.LogCaptureFixture) -> None:
    """The observer's WARNING says the call goes on unchanged: it is a call's. A synthetic check's outcome is the
    heartbeat's line, and it takes no place of the next call's warning."""
    caplog.set_level(logging.INFO, logger="collieai_guardrail")
    observer = _observer(MockServer(BUSY))
    await asyncio.wait_for(synthetic_check(observer), timeout=5)
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]
    await observer.observe_request(chat_request(), call_type="acompletion", facts=CallFacts(operation_id="op-1"))
    [warning] = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert warning.startswith("collieai guardrail pre_call: observation bypassed (busy,")


@pytest.mark.parametrize(
    ("pre", "post", "expected"),
    [
        (("bypass", "busy"), ("completed", None), True),
        (("completed", None), ("bypass", "busy"), True),
        (("bypass", "busy"), ("bypass", "busy"), True),
        (("completed", None), ("completed", None), False),
        (("bypass", "busy"), ("bypass", "auth"), False),  # a wrong key is no meeting at the server
        (("bypass", "timeout"), ("completed", None), False),
        (("outside", "busy"), ("completed", None), False),
    ],
)
def test_only_a_busy_answer_and_nothing_worse_repeats_a_synthetic_check(
    pre: tuple[str, str | None], post: tuple[str, str | None], expected: bool
) -> None:
    assert only_busy((Outcome("pre_call", *pre), Outcome("post_call", *post))) is expected


async def test_the_loop_outlives_its_own_faults(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="collieai_guardrail")
    counters = Counters()
    heartbeat = Heartbeat(
        _observer(MockServer(refuses_connection)), counters, summary_interval_s=1, synthetic_interval_s=0
    )

    def broken() -> dict[str, dict[str, object]]:
        raise RuntimeError("boom")

    monkeypatch.setattr(counters, "snapshot", broken)
    assert heartbeat.summarize() is None

    async def broken_check(observer: Observer) -> Any:
        raise RuntimeError("boom")

    monkeypatch.setattr("collieai_guardrail.heartbeat.synthetic_check", broken_check)
    assert await heartbeat.run_synthetic() is None
    messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert messages == [
        "collieai guardrail summary failed inside the adapter (RuntimeError)",
        "collieai guardrail synthetic check failed inside the adapter (RuntimeError)",
    ]
    assert "boom" not in "\n".join(messages)


async def test_a_retired_owners_loop_ends_for_good_and_releases(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="collieai_guardrail")
    server = MockServer(refuses_connection)
    released: list[bool] = []

    async def release() -> None:
        released.append(True)

    heartbeat = Heartbeat(
        _observer(server),
        Counters(),
        summary_interval_s=1,
        synthetic_interval_s=1,
        retired=lambda: True,
        on_retired=release,
    )
    heartbeat.ensure_running()
    task = heartbeat._task
    assert task is not None
    await asyncio.wait_for(task, timeout=3)
    assert released == [True] and server.requests == []  # asked before the synthetic check at start
    assert any("no longer has this instance" in r.getMessage() for r in caplog.records)
    heartbeat.ensure_running()  # a hook that still reaches the retired owner does not start it again
    assert heartbeat._task is task


async def test_a_registration_that_cannot_be_checked_keeps_the_loop(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="collieai_guardrail")
    server = MockServer(refuses_connection)

    def broken() -> bool:
        raise RuntimeError(SECRET_TEXT)

    heartbeat = Heartbeat(
        _observer(server),
        Counters(),
        summary_interval_s=3600,
        synthetic_interval_s=3600,
        retired=broken,
        start_spread_s=0,
    )
    heartbeat.ensure_running()
    try:
        for _ in range(300):
            if len(server.requests) == 2:
                break
            await asyncio.sleep(0.01)
    finally:
        await heartbeat.stop()
    assert len(server.requests) == 2  # the synthetic check ran: the loop was kept
    messages = [r.getMessage() for r in caplog.records if "registration" in r.getMessage()]
    assert messages == ["collieai guardrail: checking the registration failed (RuntimeError)"]


def test_outside_an_event_loop_the_heartbeat_waits_for_one() -> None:
    heartbeat = Heartbeat(
        _observer(MockServer(refuses_connection)), Counters(), summary_interval_s=60, synthetic_interval_s=0
    )
    heartbeat.ensure_running()  # no running loop: nothing to start, nothing raised
    assert heartbeat._task is None
