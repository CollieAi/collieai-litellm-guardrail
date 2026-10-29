"""`enforcement_mode: block` — enforce with blocking only. The adapter refuses a call the CollieAi policy blocks (400),
a call outside what it checks unless its type generates nothing a policy could block (400), and a call it could not
check (503, fail-closed); it records the phase first. Monitor stays as it was: the same replies never refuse a call."""

from __future__ import annotations

import copy
import json
import logging
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("litellm")
registry = pytest.importorskip("litellm.proxy.guardrails.guardrail_registry")

import litellm  # noqa: E402
from litellm.caching.dual_cache import DualCache  # noqa: E402
from litellm.exceptions import GuardrailRaisedException  # noqa: E402
from litellm.proxy._types import UserAPIKeyAuth  # noqa: E402
from litellm.proxy.guardrails.init_guardrails import init_guardrails_v2  # noqa: E402
from litellm.proxy.utils import ProxyLogging  # noqa: E402
from litellm.types.guardrails import GuardrailEventHooks, Mode  # noqa: E402

import collieai_guardrail.observer as observer_module  # noqa: E402
from collieai_guardrail.config import AdapterConfig, ConfigError  # noqa: E402
from collieai_guardrail.counters import Counters  # noqa: E402
from collieai_guardrail.guardrail import CollieAIGuardrail, require_guardrail  # noqa: E402
from collieai_guardrail.heartbeat import Heartbeat  # noqa: E402
from collieai_guardrail.litellm_data import (  # noqa: E402
    LoggedCalls,
    count_logged,
    count_refused,
    issue_operation_id,
    mark_refused,
)
from collieai_guardrail.observer import Observer  # noqa: E402
from collieai_guardrail.payload import CallFacts, Sender, pre_call_payload  # noqa: E402
from collieai_guardrail.verdict import Check, Rejected, parse_check  # noqa: E402
from litellm_contract import (  # noqa: E402
    API_BASE,
    API_KEY,
    MockServer,
    chat_request,
    chat_response,
    check_reply,
    enforce_reply,
    error_body,
    json_reply,
    never_answers,
)

FACTS = CallFacts(operation_id="op-1")
HANDLER = registry.IN_MEMORY_GUARDRAIL_HANDLER


def _body(reply: dict[str, Any]) -> bytes:
    return json.dumps(reply).encode()


def _observer(server: MockServer, mode: str = "block", **settings: Any) -> Observer:
    config = AdapterConfig(api_base=API_BASE, api_key=API_KEY, enforcement_mode=mode, **settings)
    return Observer(config, litellm_version="1.102.4", client=server.client())


# ------------------------------------------------------------------ the setting


def test_the_mode_is_monitor_unless_set() -> None:
    assert AdapterConfig(api_base=API_BASE, api_key=API_KEY).enforce is False
    assert AdapterConfig(api_base=API_BASE, api_key=API_KEY, enforcement_mode="block").enforce is True


@pytest.mark.parametrize("value", ["enforce", "BLOCK", "", None, 1, ["block"]])
def test_another_mode_does_not_start(value: object) -> None:
    with pytest.raises(ConfigError, match="enforcement_mode must be monitor or block"):
        AdapterConfig(api_base=API_BASE, api_key=API_KEY, enforcement_mode=value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"event_hook": ["pre_call"], "default_on": True}, "mode: \\[pre_call, post_call\\]"),
        ({"event_hook": "post_call", "default_on": True}, "mode: \\[pre_call, post_call\\]"),
        ({"event_hook": ["pre_call", "post_call"], "default_on": False}, "default_on: true"),
        ({"event_hook": ["pre_call", "post_call"]}, "default_on: true"),
    ],
)
def test_block_needs_both_phases_and_default_on(params: dict[str, Any], message: str) -> None:
    """A phase the adapter does not see, or a request that does not name it, would go through unchecked."""
    with pytest.raises(ConfigError, match=message):
        CollieAIGuardrail(api_base=API_BASE, api_key=API_KEY, enforcement_mode="block", **params)


@pytest.mark.parametrize(
    "mode",
    [Mode(tags={"lax": "pre_call"}, default=["pre_call", "post_call"]), Mode(tags={"all": ["pre_call", "post_call"]})],
    ids=["a tag with one phase", "no default"],
)
def test_block_refuses_a_mode_by_tags(mode: Mode) -> None:
    """A tag that names one phase skips the other for its calls, a mode with no default skips the untagged ones, and
    the caller sends a request's tags."""
    with pytest.raises(ConfigError, match="not a mode by tags"):
        CollieAIGuardrail(
            api_base=API_BASE, api_key=API_KEY, enforcement_mode="block", event_hook=mode, default_on=True
        )


@pytest.mark.parametrize("scope", ["non_streaming", "streaming", {"pre_call": "non_streaming"}, {"pre_call": "both"}])
async def test_block_refuses_a_stream_scope_but_both(monkeypatch: pytest.MonkeyPatch, scope: object) -> None:
    """LiteLLM's `stream_scope` runs a phase only for streaming calls or only for the others."""
    monkeypatch.setattr(litellm, "callbacks", [*litellm.callbacks, require_guardrail])
    block = CollieAIGuardrail(
        api_base=API_BASE,
        api_key=API_KEY,
        enforcement_mode="block",
        event_hook=["pre_call", "post_call"],
        default_on=True,
        synthetic_interval_s=0,
    )
    monitor = CollieAIGuardrail(
        api_base=API_BASE,
        api_key=API_KEY,
        event_hook=["pre_call", "post_call"],
        default_on=True,
        synthetic_interval_s=0,
    )
    try:
        with pytest.raises(ConfigError, match="needs stream_scope unset or both"):
            block.apply_stream_scope(scope)
        block.apply_stream_scope("both")
        block.apply_stream_scope(None)
        monitor.apply_stream_scope(scope)  # Monitor observes what LiteLLM gives it
    finally:
        for guardrail in (block, monitor):
            await guardrail.heartbeat.stop()
            await guardrail.observer.aclose()


# ------------------------------------------------------------------ the request


def test_block_asks_the_server_for_it_and_monitor_sends_what_it_sent() -> None:
    def extension(enforcement: str) -> dict[str, Any]:
        sender = Sender(adapter_version="0.3.0", litellm_version="1.102.4", budget_ms=2000, enforcement=enforcement)
        payload = pre_call_payload(chat_request(), FACTS, sender)
        return payload["additional_provider_specific_params"]["collieai"]  # type: ignore[index, return-value]

    assert extension("block")["enforcement"] == "block"
    assert "enforcement" not in extension("monitor")
    assert (extension("block")["adapter"], extension("block")["source_endpoint"]) == ("native", "chat_completions")


# ------------------------------------------------------------------ the reply


def test_a_pass_in_enforce_is_a_check() -> None:
    parsed = parse_check(_body(enforce_reply()), phase="pre_call", enforce=True)

    assert isinstance(parsed, Check) and not parsed.blocked


def test_a_block_is_a_check_that_blocked_and_names_its_rule() -> None:
    parsed = parse_check(_body(enforce_reply(blocked=True)), phase="pre_call", enforce=True)

    assert isinstance(parsed, Check)
    assert (parsed.blocked, parsed.blocked_by, parsed.categories) == (True, "rule-b", ("aho_corasick", "structured_id"))
    reasons = [f.get("no_enforce_reason") for f in parsed.diagnostics["items"][0]["findings"]]  # type: ignore[index]
    assert reasons == [None, "surface_mask_unsupported"]


def _broken(change: str) -> dict[str, Any]:
    reply = enforce_reply(blocked=change.startswith("blocked:"))
    collieai = reply["collieai"]
    what = change.split(":", 1)[1]
    if what == "mode monitor":
        collieai["mode"] = "monitor"
    elif what == "intervened":
        reply["action"] = "GUARDRAIL_INTERVENED"
    elif what == "no action":
        del reply["action"]
    elif what == "texts":
        reply["texts"] = ["rewritten"]
    elif what == "blocked_reason":
        reply["blocked_reason"] = "x"
    elif what == "terminal":
        collieai["terminal"] = "completed" if reply["action"] == "BLOCKED" else "blocked"
    elif what == "none judged":
        collieai["coverage"]["items_judged"] = 0
    elif what == "incomplete":
        collieai["coverage"]["state"] = "incomplete"
    elif what == "no kind":
        for item in collieai.get("items", []):
            for finding in item["findings"]:
                finding.pop("kind")
    elif what == "no acting block":
        for finding in collieai["items"][0]["findings"]:
            finding["mode"] = "monitor"
    elif what == "phase":
        collieai["phase"] = "post_call"
    return reply


@pytest.mark.parametrize(
    ("change", "detail"),
    [
        ("pass:mode monitor", "mode"),  # an old server: it would quietly lower the protection
        ("blocked:mode monitor", "mode"),
        ("pass:intervened", "action"),
        ("pass:no action", "action"),
        ("pass:texts", "replacement_key"),
        ("blocked:texts", "replacement_key"),
        ("pass:blocked_reason", "replacement_key"),
        ("pass:terminal", "terminal"),
        ("blocked:terminal", "terminal"),
        ("blocked:none judged", "items"),
        ("pass:incomplete", "coverage_state"),
        ("blocked:no kind", "finding_kind"),
        ("blocked:no acting block", "block_finding"),
        ("blocked:phase", "phase"),
    ],
)
def test_anything_else_is_a_protocol_error(change: str, detail: str) -> None:
    parsed = parse_check(_body(_broken(change)), phase="pre_call", enforce=True)

    assert parsed == Rejected(detail)


def test_monitor_never_takes_a_block() -> None:
    assert parse_check(_body(enforce_reply(blocked=True)), phase="pre_call") == Rejected("action")
    assert parse_check(_body(enforce_reply()), phase="pre_call") == Rejected("mode")


# ------------------------------------------------------------------ the outcome


async def test_a_block_refuses_the_call_before_the_model() -> None:
    server = MockServer(json_reply(200, enforce_reply(blocked=True)))

    outcome = await _observer(server).observe_request(chat_request(), call_type="acompletion", facts=FACTS)

    assert outcome.refusal == (400, "content_blocked")
    assert (outcome.guardrail_status, outcome.guardrail_action) == ("guardrail_intervened", "BLOCKED")
    assert outcome.violation_categories == ["aho_corasick", "structured_id"]
    assert outcome.diagnostics()["effective"] == "block"
    assert server.bodies[0]["additional_provider_specific_params"]["collieai"]["enforcement"] == "block"


async def test_a_block_after_the_model_names_the_response() -> None:
    server = MockServer(json_reply(200, enforce_reply("post_call", 1, blocked=True)))

    outcome = await _observer(server).observe_response(
        chat_request(), chat_response(), call_type="acompletion", facts=FACTS
    )

    assert outcome.refusal == (400, "response_blocked")


async def test_a_pass_goes_on() -> None:
    outcome = await _observer(MockServer(json_reply(200, enforce_reply()))).observe_request(
        chat_request(), call_type="acompletion", facts=FACTS
    )

    assert (outcome.refusal, outcome.guardrail_action) == (None, "NONE")


@pytest.mark.parametrize(
    "respond",
    [
        json_reply(503, error_body("busy", "busy")),
        json_reply(504, error_body("deadline", "deadline_expired")),
        json_reply(422, error_body("capability", "enforce_disabled")),
        json_reply(422, error_body("capability", "policy_not_enforce")),
        json_reply(500, error_body("internal", "invariant_violated")),
        json_reply(200, check_reply()),  # a server in monitor
        never_answers,
    ],
    ids=["busy", "deadline", "enforce_disabled", "policy_not_enforce", "server_error", "monitor_reply", "timeout"],
)
async def test_a_check_not_made_refuses_the_call(respond: Any) -> None:
    observer = _observer(MockServer(respond), deadline_ms=300, network_margin_ms=100)

    outcome = await observer.observe_request(chat_request(), call_type="acompletion", facts=FACTS)

    assert outcome.result == "bypass"
    assert outcome.refusal == (503, "guardrail_unavailable")
    assert (outcome.guardrail_status, outcome.guardrail_action) == ("guardrail_failed_to_respond", "BLOCKED")


@pytest.mark.parametrize(
    "respond",
    [
        json_reply(413, error_body("limit", "body_too_large")),
        json_reply(422, error_body("limit", "max_total_chars")),
        json_reply(422, error_body("shape", "unsupported_text_shape")),
    ],
    ids=["body_too_large", "text_limit", "shape"],
)
async def test_a_request_the_profile_does_not_take_is_a_400_not_to_retry(respond: Any) -> None:
    """The same call meets its limit or shape again: refused as outside what the adapter checks, not as a check that
    may be retried; the bypass keeps its reason."""
    observer = _observer(MockServer(respond))

    before = await observer.observe_request(chat_request(), call_type="acompletion", facts=FACTS)
    after = await observer.observe_response(chat_request(), chat_response(), call_type="acompletion", facts=FACTS)

    assert (before.result, before.refusal) == ("bypass", (400, "unsupported_request"))
    assert (after.result, after.refusal) == ("bypass", (400, "unsupported_response"))
    assert (before.guardrail_status, before.guardrail_action) == ("guardrail_intervened", "BLOCKED")
    snapshot = observer.counters.snapshot()
    assert snapshot["pre_call"]["refused"] == {"unsupported_request": 1}
    assert snapshot["pre_call"]["bypass"] in ({"limit": 1}, {"shape": 1})


async def test_a_body_over_its_cap_is_refused_unsent_with_a_400() -> None:
    server = MockServer(json_reply(200, enforce_reply()))
    observer = _observer(server, max_body_bytes=1024)

    outcome = await observer.observe_request(
        chat_request(messages=[{"role": "user", "content": "x" * 2000}]), call_type="acompletion", facts=FACTS
    )

    assert (outcome.reason, outcome.detail, outcome.refusal) == (
        "limit",
        "body_over_limit",
        (400, "unsupported_request"),
    )
    assert server.requests == []


async def test_monitor_never_refuses_a_request_over_the_limits() -> None:
    observer = _observer(MockServer(json_reply(413, error_body("limit", "body_too_large"))), mode="monitor")

    outcome = await observer.observe_request(chat_request(), call_type="acompletion", facts=FACTS)

    assert (outcome.result, outcome.reason, outcome.refusal) == ("bypass", "limit", None)


@pytest.mark.parametrize(
    ("request_", "call_type"),
    [
        (chat_request(stream=True), "acompletion"),
        (chat_request(tools=[{"type": "function", "function": {"name": "f"}}]), "acompletion"),
        (chat_request(), "aresponses"),
        (chat_request(), "anthropic_messages"),
        (chat_request(), "pass_through_endpoint"),
        (chat_request(), "a_type_litellm_adds_later"),
        (chat_request(), None),
    ],
    ids=["stream", "tools", "responses", "messages", "pass_through", "unknown", "no_type"],
)
async def test_a_call_outside_the_profile_is_refused_unsent(request_: dict[str, Any], call_type: str | None) -> None:
    server = MockServer(json_reply(200, enforce_reply()))

    outcome = await _observer(server).observe_request(request_, call_type=call_type, facts=FACTS)

    assert outcome.refusal == (400, "unsupported_request")
    assert outcome.guardrail_status == "guardrail_intervened"
    assert server.requests == []


@pytest.mark.parametrize("call_type", ["aembedding", "embedding", "arerank", "rerank", "amoderation", "moderation"])
async def test_a_call_that_generates_nothing_passes_unchecked(call_type: str) -> None:
    outcome = await _observer(MockServer(json_reply(200, enforce_reply()))).observe_request(
        chat_request(), call_type=call_type, facts=FACTS
    )

    assert (outcome.result, outcome.refusal, outcome.guardrail_status) == ("outside", None, "not_run")


async def test_a_response_outside_the_profile_is_refused() -> None:
    response = chat_response(reasoning_content="thinking")

    outcome = await _observer(MockServer(json_reply(200, enforce_reply("post_call", 1)))).observe_response(
        chat_request(), response, call_type="acompletion", facts=FACTS
    )

    assert outcome.refusal == (400, "unsupported_response")


@pytest.mark.parametrize(
    "reply", [enforce_reply(blocked=True), error_body("busy", "busy")], ids=["blocked_reply", "busy"]
)
async def test_monitor_never_refuses(reply: dict[str, Any]) -> None:
    status = 200 if "action" in reply else 503
    outcome = await _observer(MockServer(json_reply(status, reply)), mode="monitor").observe_request(
        chat_request(), call_type="acompletion", facts=FACTS
    )
    outside = await _observer(MockServer(json_reply(200, reply)), mode="monitor").observe_request(
        chat_request(stream=True), call_type="acompletion", facts=FACTS
    )

    assert (outcome.refusal, outside.refusal) == (None, None)


async def test_the_counts_name_blocks_and_refusals() -> None:
    counters = Counters()
    observer = _observer(MockServer(json_reply(200, enforce_reply(blocked=True))))
    observer.counters = counters

    await observer.observe_request(chat_request(), call_type="acompletion", facts=FACTS)
    await observer.observe_request(chat_request(stream=True), call_type="acompletion", facts=FACTS)
    snapshot = counters.snapshot()["pre_call"]

    assert snapshot["blocked"] == 1
    assert snapshot["refused"] == {"unsupported_request": 1}


# ------------------------------------------------------------------ the glue, in LiteLLM's own dispatch


def _entry(**params: Any) -> dict[str, Any]:
    return {
        "guardrail_name": "collieai-block",
        "litellm_params": {
            "guardrail": "collieai_guardrail.guardrail.CollieAIGuardrail",
            "mode": ["pre_call", "post_call"],
            "default_on": True,
            "api_base": API_BASE,
            "api_key": API_KEY,
            "enforcement_mode": "block",
            "synthetic_interval_s": 0,
            "summary_interval_s": 3600,
            **params,
        },
    }


def _forget_ours() -> list[CollieAIGuardrail]:
    ours = {id(c): c for c in litellm.callbacks if isinstance(c, CollieAIGuardrail)}
    for guardrail_id, callback in list(HANDLER.guardrail_id_to_custom_guardrail.items()):
        if isinstance(callback, CollieAIGuardrail):
            HANDLER.delete_in_memory_guardrail(guardrail_id)
            ours[id(callback)] = callback
    for callback in ours.values():
        litellm.logging_callback_manager.remove_callback_from_all_lists(callback)
    return list(ours.values())


@pytest.fixture
async def loaded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[Any]:
    _forget_ours()
    monkeypatch.setattr(litellm, "callbacks", [*litellm.callbacks, require_guardrail])  # block needs the guard

    def make(server: MockServer, **params: Any) -> CollieAIGuardrail:
        monkeypatch.setattr(observer_module, "new_client", server.client)
        init_guardrails_v2(all_guardrails=[_entry(**params)], config_file_path=str(tmp_path / "config.yaml"))
        [guardrail] = [c for c in litellm.callbacks if isinstance(c, CollieAIGuardrail)]
        return guardrail

    yield make
    for guardrail in _forget_ours():
        await guardrail.heartbeat.stop()
        await guardrail.observer.aclose()


def _guardrail_records(data: dict[str, Any]) -> list[dict[str, Any]]:
    return list(data["metadata"]["standard_logging_guardrail_information"])


async def test_litellms_dispatch_refuses_a_blocked_call_and_the_record_says_so(loaded: Any) -> None:
    loaded(MockServer(json_reply(200, enforce_reply(blocked=True))))
    data = copy.deepcopy(chat_request())

    with pytest.raises(GuardrailRaisedException) as caught:
        await ProxyLogging(user_api_key_cache=DualCache()).pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(request_route="/v1/chat/completions"), data=data, call_type="acompletion"
        )

    assert caught.value.status_code == 400
    assert "(content_blocked)" in str(caught.value) and "4111" not in str(caught.value)
    [record] = _guardrail_records(data)
    assert (record["guardrail_status"], record["guardrail_action"]) == ("guardrail_intervened", "BLOCKED")


@pytest.mark.parametrize("scope", ["non_streaming", {"pre_call": "non_streaming"}])
async def test_a_stream_scope_never_leaves_a_streaming_call_unchecked(loaded: Any, scope: object) -> None:
    """Through LiteLLM's loader: where LiteLLM has `stream_scope` (1.106) the proxy does not start with it; where it
    has none (1.102) the guardrail still runs for a streaming call, which block refuses as outside what it checks."""
    try:
        guardrail = loaded(MockServer(json_reply(200, enforce_reply())), stream_scope=scope)
    except ConfigError:
        return
    streaming = {**copy.deepcopy(chat_request()), "stream": True}
    assert guardrail.should_run_guardrail(streaming, GuardrailEventHooks.pre_call)


async def test_a_check_not_made_is_a_503(loaded: Any) -> None:
    loaded(MockServer(json_reply(503, error_body("busy", "busy"))))

    with pytest.raises(GuardrailRaisedException) as caught:
        await ProxyLogging(user_api_key_cache=DualCache()).pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(request_route="/v1/chat/completions"),
            data=copy.deepcopy(chat_request()),
            call_type="acompletion",
        )

    assert caught.value.status_code == 503


async def test_a_fault_of_the_glue_refuses_the_call(loaded: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    guardrail = loaded(MockServer(json_reply(200, enforce_reply())))

    def broken(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("glue")

    monkeypatch.setattr(guardrail, "_record", broken)

    with pytest.raises(GuardrailRaisedException) as caught:
        await guardrail.async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(request_route="/v1/chat/completions"),
            cache=None,
            data=copy.deepcopy(chat_request()),
            call_type="acompletion",
        )

    assert caught.value.status_code == 503


async def test_a_blocked_response_is_refused_after_the_model(loaded: Any) -> None:
    guardrail = loaded(MockServer(json_reply(200, enforce_reply("post_call", 1, blocked=True))))

    with pytest.raises(GuardrailRaisedException) as caught:
        await guardrail.async_post_call_success_hook(
            data=copy.deepcopy(chat_request()),
            user_api_key_dict=UserAPIKeyAuth(request_route="/v1/chat/completions"),
            response=chat_response(),
        )

    assert caught.value.status_code == 400 and "(response_blocked)" in str(caught.value)


async def test_a_pass_returns_none(loaded: Any) -> None:
    guardrail = loaded(MockServer(json_reply(200, enforce_reply())))

    assert (
        await guardrail.async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(request_route="/v1/chat/completions"),
            cache=None,
            data=copy.deepcopy(chat_request()),
            call_type="acompletion",
        )
        is None
    )


# ------------------------------------------------------------------ the count apart from the hooks


def _logged(metadata: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    return {
        "call_type": "acompletion",
        "messages": [{"role": "user", "content": "hi"}],
        "litellm_params": {"metadata": metadata},
        **kwargs,
    }


def _logged_counts(counters: Counters) -> tuple[Any, ...]:
    """Per phase: logged calls, of them eligible, and further attempts."""
    snapshot = counters.snapshot()
    return tuple(
        (snapshot[p]["logged_calls"], snapshot[p]["logged_eligible"], snapshot[p]["logged_repeats"])
        for p in ("pre_call", "post_call")
    )


@pytest.mark.parametrize("event", ["none", "rejected", "failed"])
@pytest.mark.parametrize("phase", ["pre_call", "post_call"])
def test_our_own_refusal_is_counted_once_whatever_litellm_logs(phase: str, event: str) -> None:
    """LiteLLM logs a guardrail's refusal by its version: not at all, or as a call its proxy refused (also after the
    model), or as a failed one. The refusal is counted when it is made, and the event about it, if any, skipped."""
    counters, calls = Counters(), LoggedCalls()
    metadata: dict[str, Any] = {}
    operation_id = issue_operation_id(metadata)
    mark_refused(metadata, phase, operation_id, None)
    count_refused(counters, phase, None, operation_id=operation_id, calls=calls)

    if event != "none":
        extra = {"litellm_no_upstream_llm_call": True} if event == "rejected" else {}
        count_logged(counters, _logged(metadata, **extra), RuntimeError("refused"), succeeded=False, calls=calls)

    assert _logged_counts(counters) == ((1, 1, 0), (1, 1, 0) if phase == "post_call" else (0, 0, 0))


def test_a_refusal_after_the_model_counts_the_pre_call_phase_by_the_request() -> None:
    counters = Counters()
    metadata: dict[str, Any] = {}
    operation_id = issue_operation_id(metadata)

    count_refused(counters, "post_call", "reasoning", operation_id=operation_id, request_reason="call_type")

    snapshot = counters.snapshot()
    assert (snapshot["pre_call"]["logged_outside"], snapshot["post_call"]["logged_outside"]) == (
        {"call_type": 1},
        {"reasoning": 1},
    )


def test_a_refusal_after_a_logged_attempt_counts_the_pre_call_phase_once() -> None:
    """The router logged a failed attempt of the call before the one the guardrail refused after the model."""
    counters, calls = Counters(), LoggedCalls()
    metadata: dict[str, Any] = {}
    operation_id = issue_operation_id(metadata)
    count_logged(counters, _logged(metadata), RuntimeError("attempt"), succeeded=False, calls=calls)

    mark_refused(metadata, "post_call", operation_id, None)
    count_refused(counters, "post_call", None, operation_id=operation_id, calls=calls)

    assert _logged_counts(counters) == ((1, 1, 0), (1, 1, 0))


def test_a_mark_the_caller_wrote_is_not_taken() -> None:
    counters = Counters()
    metadata: dict[str, Any] = {}
    issue_operation_id(metadata)
    metadata["collieai_pre_refused"] = ":" + "0" * 32

    count_logged(counters, _logged(metadata, litellm_no_upstream_llm_call=True), None, succeeded=False)

    assert counters.snapshot()["pre_call"]["logged_outside"] == {"rejected": 1}


# ------------------------------------------------------------------ the synthetic check


async def test_a_failed_synthetic_check_in_block_is_an_error(caplog: pytest.LogCaptureFixture) -> None:
    observer = _observer(MockServer(json_reply(422, error_body("capability", "enforce_disabled"))))
    heartbeat = Heartbeat(
        observer, Counters(), summary_interval_s=3600, synthetic_interval_s=0, retired=lambda: False, name="g"
    )

    with caplog.at_level(logging.INFO, logger="collieai_guardrail"):
        await heartbeat.run_synthetic()

    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1 and "calls are likely being refused" in errors[0].getMessage()
    await observer.aclose()


@pytest.fixture(autouse=True)
def _no_leftover_guardrails() -> Iterator[None]:
    yield
    for callback in [c for c in litellm.callbacks if isinstance(c, CollieAIGuardrail)]:
        litellm.logging_callback_manager.remove_callback_from_all_lists(callback)


# ------------------------------------------------------------------ output, marks and faults


async def test_a_call_that_asks_for_audio_is_refused_before_the_model() -> None:
    """Post-call could not check an audio reply, so block refuses the call before the model; Monitor still checks
    the request's text."""
    request = chat_request(modalities=["text", "audio"], audio={"voice": "alloy", "format": "wav"})
    blocking = MockServer(json_reply(200, enforce_reply()))
    observing = MockServer(json_reply(200, check_reply()))

    refused = await _observer(blocking).observe_request(request, call_type="acompletion", facts=FACTS)
    observed = await _observer(observing, mode="monitor").observe_request(request, call_type="acompletion", facts=FACTS)

    assert (refused.refusal, refused.reason, blocking.requests) == ((400, "unsupported_request"), "non_text_output", [])
    assert (observed.result, observed.refusal, len(observing.requests)) == ("completed", None, 1)


@pytest.mark.parametrize("modalities", [["audio"], ["text", "image"], "text"])
async def test_any_output_but_text_is_refused(modalities: object) -> None:
    outcome = await _observer(MockServer(json_reply(200, enforce_reply()))).observe_request(
        chat_request(modalities=modalities), call_type="acompletion", facts=FACTS
    )

    assert outcome.refusal == (400, "unsupported_request")


async def test_a_text_only_modality_is_checked() -> None:
    outcome = await _observer(MockServer(json_reply(200, enforce_reply()))).observe_request(
        chat_request(modalities=["text"]), call_type="acompletion", facts=FACTS
    )

    assert (outcome.result, outcome.refusal) == ("completed", None)


@pytest.mark.parametrize("value", ["привет", "x y", "a" * 65])
def test_a_mark_of_another_form_is_ignored_not_an_error(value: str) -> None:
    """A caller's mark that is no closed reason never reaches the tag; the event is counted as any other."""
    counters = Counters()
    metadata: dict[str, Any] = {}
    issue_operation_id(metadata)
    metadata["collieai_post_refused"] = f"{value}:" + "0" * 32

    count_logged(counters, _logged(metadata, litellm_no_upstream_llm_call=True), None, succeeded=False)

    assert counters.snapshot()["pre_call"]["logged_outside"] == {"rejected": 1}


async def test_a_synthetic_check_that_fails_inside_the_adapter_is_an_error_in_block(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    import collieai_guardrail.heartbeat as heartbeat_module

    async def broken(observer: Any) -> Any:
        raise RuntimeError("inside")

    monkeypatch.setattr(heartbeat_module, "synthetic_check", broken)
    observer = _observer(MockServer(json_reply(200, enforce_reply())))
    heartbeat = Heartbeat(
        observer, Counters(), summary_interval_s=3600, synthetic_interval_s=0, retired=lambda: False, name="g"
    )

    with caplog.at_level(logging.INFO, logger="collieai_guardrail"):
        await heartbeat.run_synthetic()

    assert [r.levelname for r in caplog.records if "synthetic check" in r.getMessage()] == ["WARNING", "ERROR"]
    await observer.aclose()


async def test_a_fault_of_the_glue_marks_its_refusal(loaded: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    guardrail = loaded(MockServer(json_reply(200, enforce_reply())))
    data = copy.deepcopy(chat_request())

    def broken(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("glue")

    monkeypatch.setattr(guardrail, "_record", broken)
    with pytest.raises(GuardrailRaisedException):
        await guardrail.async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(request_route="/v1/chat/completions"),
            cache=None,
            data=data,
            call_type="acompletion",
        )
    guardrail._count_logged(_logged(data["metadata"], litellm_no_upstream_llm_call=True), None, succeeded=False)

    assert _logged_counts(guardrail.counters) == ((1, 1, 0), (0, 0, 0))


async def test_a_refusal_after_the_model_counts_an_outside_request_by_its_reason(
    loaded: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fault of the glue after the model of a call outside by its type: its pre-call phase counts under that type."""
    guardrail = loaded(MockServer(json_reply(200, enforce_reply())))
    data = copy.deepcopy(chat_request())
    issue_operation_id(data["metadata"])

    def broken(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("glue")

    monkeypatch.setattr(guardrail, "_record", broken)
    with pytest.raises(GuardrailRaisedException):
        await guardrail.async_post_call_success_hook(
            data=data, user_api_key_dict=UserAPIKeyAuth(request_route="/v1/embeddings"), response=chat_response()
        )

    assert guardrail.counters.snapshot()["pre_call"]["logged_outside"] == {"call_type": 1}


@pytest.mark.parametrize("phase", ["pre_call", "post_call"])
async def test_a_fault_of_the_glue_keeps_the_outside_reason(
    phase: str, loaded: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An outside call whose record failed is counted under its reason, not as an eligible one."""
    guardrail = loaded(MockServer(json_reply(200, enforce_reply())))
    data = copy.deepcopy(chat_request(modalities=["text", "audio"]) if phase == "pre_call" else chat_request())

    def broken(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("glue")

    monkeypatch.setattr(guardrail, "_record", broken)
    auth = UserAPIKeyAuth(request_route="/v1/chat/completions")
    with pytest.raises(GuardrailRaisedException):
        if phase == "pre_call":
            await guardrail.async_pre_call_hook(user_api_key_dict=auth, cache=None, data=data, call_type="acompletion")
        else:
            issue_operation_id(data["metadata"])
            await guardrail.async_post_call_success_hook(
                data=data, user_api_key_dict=auth, response=chat_response(reasoning_content="thinking")
            )
    logged = _logged(data["metadata"], **({"litellm_no_upstream_llm_call": True} if phase == "pre_call" else {}))
    guardrail._count_logged(logged, RuntimeError("refused"), succeeded=False)

    snapshot = guardrail.counters.snapshot()
    want = "non_text_output" if phase == "pre_call" else "reasoning"
    assert (snapshot[phase]["logged_eligible"], snapshot[phase]["logged_outside"]) == (0, {want: 1})
    if phase == "post_call":
        assert (snapshot["pre_call"]["logged_eligible"], snapshot["pre_call"]["logged_outside"]) == (1, {})
