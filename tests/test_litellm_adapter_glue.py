"""The LiteLLM glue against LiteLLM's own code and objects (1.102.4). Skipped where LiteLLM is not installed. The
proxy's HTTP server itself, with its hook dispatch, is not exercised here."""

from __future__ import annotations

import asyncio
import copy
import datetime
import io
import json
import logging
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

litellm = pytest.importorskip("litellm")

from litellm.integrations.custom_guardrail import CustomGuardrail  # noqa: E402
from litellm.litellm_core_utils.litellm_logging import Logging  # noqa: E402
from litellm.proxy._types import UserAPIKeyAuth  # noqa: E402
from litellm.proxy.types_utils.utils import get_instance_fn  # noqa: E402
from litellm.types.guardrails import GuardrailEventHooks, LitellmParams, Mode  # noqa: E402
from litellm.types.utils import Choices, Message, ModelResponse  # noqa: E402

import collieai_guardrail.observer as observer_module  # noqa: E402
from collieai_guardrail.config import ConfigError  # noqa: E402
from collieai_guardrail.guardrail import CollieAIGuardrail  # noqa: E402
from collieai_guardrail.litellm_data import OPERATION_ID_FORM, OPERATION_ID_KEY, OPERATION_TAG_KEY  # noqa: E402
from litellm_contract import (  # noqa: E402
    API_BASE,
    API_KEY,
    SECRET_TEXT,
    MockServer,
    chat_request,
    check_reply,
    json_reply,
    never_answers,
)

#: The pinned LiteLLM's own types warn under pydantic when it first builds a reasoning item's schema; that warning is
#: LiteLLM's, the suite runs with `-W error` for everything else.
pytestmark = pytest.mark.filterwarnings(
    "ignore:Item 'summary' on TypedDict class 'ChatCompletionReasoningItem':UserWarning"
)

GUARDRAIL_PATH = "collieai_guardrail.guardrail.CollieAIGuardrail"
ROUTE = "/v1/chat/completions"
Build = Callable[..., CollieAIGuardrail]


@pytest.fixture
async def build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[Build]:
    """The guardrail as LiteLLM's registry builds it (`GuardrailRegistry.initialize_custom_guardrail`, which needs
    the proxy's FastAPI to import): the class from the config's path through LiteLLM's loader, `litellm_params`
    dumped without None and without `guardrail`, `mode`, `default_on`, and the result added to LiteLLM's callbacks.
    The adapter's own client is swapped for the mock server's."""
    built: list[CollieAIGuardrail] = []

    def make(server: MockServer, *, mode: Any = ("pre_call", "post_call"), default_on: bool = True, **params: Any):
        monkeypatch.setattr(observer_module, "new_client", server.client)
        litellm_params = LitellmParams(
            guardrail=GUARDRAIL_PATH,
            mode=list(mode) if isinstance(mode, tuple) else mode,
            default_on=default_on,
            **{
                "api_base": API_BASE,
                "api_key": API_KEY,
                "deadline_ms": 1000,
                "network_margin_ms": 100,
                "synthetic_interval_s": 0,
                "summary_interval_s": 3600,
                **params,
            },
        )
        guardrail_class = get_instance_fn(GUARDRAIL_PATH, config_file_path=str(tmp_path / "config.yaml"))
        extra = litellm_params.model_dump(exclude_none=True)
        for key in ("guardrail", "mode", "default_on"):
            extra.pop(key, None)
        guardrail = guardrail_class(
            guardrail_name="collieai-monitor",
            event_hook=litellm_params.mode,
            default_on=litellm_params.default_on,
            **extra,
        )
        litellm.logging_callback_manager.add_litellm_callback(guardrail)
        built.append(guardrail)
        return guardrail

    yield make
    for guardrail in built:
        litellm.logging_callback_manager.remove_callback_from_all_lists(guardrail)
        await guardrail.heartbeat.stop()
        await guardrail.observer.aclose()


def _server() -> MockServer:
    """Answers each phase with the route's example reply for its phase."""

    def respond(request: Any) -> Any:
        phase = "pre_call" if json.loads(request.content)["input_type"] == "request" else "post_call"
        return json_reply(200, check_reply(phase, items=3 if phase == "pre_call" else 1))(request)

    return MockServer(respond)


def _proxy_data(**extra: Any) -> dict[str, Any]:
    """A chat request as the proxy hands it to the hooks: with LiteLLM's logging object and its metadata."""
    request = chat_request(**extra)
    request["litellm_logging_obj"] = Logging(
        model="gpt-4o-mini",
        messages=request["messages"],
        stream=False,
        call_type="acompletion",
        start_time=datetime.datetime.now(),
        litellm_call_id="call-1",
        function_id="f-1",
    )
    request["proxy_server_request"] = {"url": "http://proxy" + ROUTE, "headers": {"x-collieai-conversation-id": "c-9"}}
    return request


def _records(data: dict[str, Any]) -> list[dict[str, Any]]:
    return list(data["metadata"].get("standard_logging_guardrail_information") or [])


async def _reply(text: str = "the reply") -> ModelResponse:
    response = await litellm.acompletion(
        model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}], mock_response=text
    )
    assert isinstance(response, ModelResponse)
    return response


def _user(route: str | None = ROUTE) -> UserAPIKeyAuth:
    return UserAPIKeyAuth(request_route=route)


# ------------------------------------------------------------------ loading


async def test_litellm_builds_the_guardrail_from_its_config_as_its_registry_does(build: Build) -> None:
    guardrail = build(_server(), deadline_ms=2500, network_margin_ms=500, forward_conversation_id=True)
    assert isinstance(guardrail, CustomGuardrail)
    assert guardrail.guardrail_name == "collieai-monitor" and guardrail.default_on is True
    assert guardrail.config.budget_ms == 2000 and guardrail.config.forward_conversation_id is True
    assert guardrail.config.synthetic_interval_s == 0
    assert guardrail.supported_event_hooks == [GuardrailEventHooks.pre_call, GuardrailEventHooks.post_call]
    assert guardrail in litellm.callbacks
    assert "apply_guardrail" not in CollieAIGuardrail.__dict__  # so the proxy calls the hooks themselves
    assert guardrail.heartbeat._task is not None  # started on the loop LiteLLM built it on


@pytest.mark.parametrize("api_key", [None, "None"])  # unset: None from the proxy's config, "None" from the registry
async def test_a_missing_key_stops_the_start_naming_the_setting_not_a_value(build: Build, api_key: Any) -> None:
    with pytest.raises(ConfigError, match="api_key"):
        build(_server(), api_key=api_key)
    with pytest.raises(ConfigError) as refused:
        build(_server(), deadline_ms=2500, network_margin_ms=2500)
    assert API_KEY not in str(refused.value)


@pytest.mark.parametrize("strict", [None, "false"])
@pytest.mark.parametrize(
    "mode",
    [
        "during_call",
        "logging_only",
        ["pre_call", "during_call"],
        [],
        Mode(tags={"team-a": "during_call"}),
        Mode(tags={"team-a": "pre_call"}, default="during_call"),
    ],
)
async def test_a_mode_beyond_pre_and_post_call_is_refused_at_start_whatever_litellms_strictness(
    build: Build, monkeypatch: pytest.MonkeyPatch, mode: Any, strict: str | None
) -> None:
    """Under LITELLM_STRICT_GUARDRAIL_MODES=false LiteLLM only warns about a mode its guardrail does not support."""
    if strict is None:
        monkeypatch.delenv("LITELLM_STRICT_GUARDRAIL_MODES", raising=False)
    else:
        monkeypatch.setenv("LITELLM_STRICT_GUARDRAIL_MODES", strict)
    with pytest.raises(ConfigError, match="mode") as refused:
        build(_server(), mode=mode)
    assert not isinstance(refused.value, (ValueError, TypeError))  # what LiteLLM's loader would swallow
    assert all(not isinstance(callback, CollieAIGuardrail) for callback in litellm.callbacks)


@pytest.mark.parametrize(
    ("mode", "unobserved"),
    [
        (("pre_call", "post_call"), None),
        ("post_call", "pre_call"),
        (Mode(tags={"team-a": ["pre_call", "post_call"]}), None),
        (Mode(tags={"team-a": "pre_call"}, default="post_call"), None),
        (Mode(tags={"team-a": "pre_call"}), "post_call"),
    ],
)
async def test_a_mode_of_its_own_hooks_loads_and_warns_of_a_phase_it_leaves_out(
    build: Build, caplog: pytest.LogCaptureFixture, mode: Any, unobserved: str | None
) -> None:
    caplog.set_level(logging.WARNING, logger="collieai_guardrail")
    guardrail = build(_server(), mode=mode)
    assert guardrail in litellm.callbacks
    warned = [r.getMessage() for r in caplog.records if "mode has no" in r.getMessage()]
    expected = (
        f"collieai guardrail: mode has no {unobserved}; that phase is not observed (set mode: [pre_call, post_call])"
    )
    assert warned == ([] if unobserved is None else [expected])


async def test_whatever_litellms_own_setup_refuses_stops_the_start_too(
    build: Build, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(self: Any, **kwargs: Any) -> None:
        raise TypeError("a refusal of LiteLLM's own")

    monkeypatch.setattr(CustomGuardrail, "__init__", refuse)
    with pytest.raises(ConfigError, match=r"refused \(TypeError\)"):
        build(_server())


async def test_an_instance_litellm_drops_from_its_callbacks_retires_and_closes_its_client(build: Build) -> None:
    """The registry's deletion and rebuild take the old instance out of every callback list
    (`remove_callback_from_all_lists`) and close nothing; the instance notices and lets go itself."""
    guardrail = build(_server(), deadline_ms=50, network_margin_ms=10)
    assert guardrail._retired() is False
    litellm.logging_callback_manager.remove_callback_from_all_lists(guardrail)
    assert guardrail._retired() is True
    release = asyncio.ensure_future(guardrail._release())
    await asyncio.sleep(0)
    assert not guardrail.observer._client.is_closed  # a hook already dispatched to it gets its deadline
    await asyncio.wait_for(release, timeout=3)
    assert guardrail.observer._client.is_closed


async def test_an_instance_removed_before_its_heartbeats_first_turn_retires_too(build: Build) -> None:
    """The registry builds and registers in one step; a removal that comes before the heartbeat ever looked still
    retires the instance that built its own client, before its first synthetic check."""
    server = _server()
    guardrail = build(server, deadline_ms=50, network_margin_ms=10, synthetic_interval_s=1)
    litellm.logging_callback_manager.remove_callback_from_all_lists(guardrail)  # no turn of the loop in between
    task = guardrail.heartbeat._task
    assert task is not None and guardrail._registered is False
    await asyncio.wait_for(task, timeout=3)
    assert guardrail.observer._client.is_closed
    assert server.requests == []


def _embedded(client: Any, **params: Any) -> CollieAIGuardrail:
    values: dict[str, Any] = {
        "api_base": API_BASE,
        "api_key": API_KEY,
        "deadline_ms": 50,
        "network_margin_ms": 10,
        "synthetic_interval_s": 0,
        "client": client,
        "guardrail_name": "embedded",
        "event_hook": ["pre_call", "post_call"],
        "default_on": True,
        **params,
    }
    return CollieAIGuardrail(**values)


async def test_an_instance_litellm_never_registered_never_retires() -> None:
    guardrail = _embedded(_server().client())
    try:
        assert guardrail._retired() is False
    finally:
        await guardrail.heartbeat.stop()
        await guardrail.observer.aclose()


async def test_a_retiring_instance_leaves_a_client_it_was_given_to_its_owner() -> None:
    client = _server().client()
    guardrail = _embedded(client)
    litellm.logging_callback_manager.add_litellm_callback(guardrail)
    try:
        litellm.logging_callback_manager.remove_callback_from_all_lists(guardrail)
        assert guardrail._retired() is False  # never seen registered: the heartbeat has not looked yet
        litellm.logging_callback_manager.add_litellm_callback(guardrail)
        assert guardrail._retired() is False
        litellm.logging_callback_manager.remove_callback_from_all_lists(guardrail)
        assert guardrail._retired() is True
        await asyncio.wait_for(guardrail._release(), timeout=3)
        assert not client.is_closed
    finally:
        litellm.logging_callback_manager.remove_callback_from_all_lists(guardrail)
        await guardrail.heartbeat.stop()
        await client.aclose()


async def test_an_embedded_instance_without_a_mode_is_on_both_phases_as_litellm_runs_it(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """LiteLLM runs a guardrail without `event_hook` on every hook (`_event_hook_is_event_type`)."""
    caplog.set_level(logging.WARNING, logger="collieai_guardrail")
    client = _server().client()
    guardrail = _embedded(client, event_hook=None)
    try:
        assert not any("mode has no" in r.getMessage() for r in caplog.records)
    finally:
        await guardrail.heartbeat.stop()
        await client.aclose()


# ------------------------------------------------------------------ its log lines on a stock proxy


def _stock_logging(monkeypatch: pytest.MonkeyPatch) -> logging.Logger:
    """Logging as the stock proxy leaves it: handlers on LiteLLM's own loggers only, the root at WARNING without one,
    the adapter's logger untouched (the suite's conftest gives it back after the test). Called in the test's body:
    pytest gives the root its own handlers as each phase of a test begins."""
    monkeypatch.setattr(logging.getLogger(), "handlers", [])
    log = logging.getLogger("collieai_guardrail")
    log.handlers.clear()
    log.setLevel(logging.NOTSET)
    return log


async def test_on_a_stock_proxy_the_adapter_logs_its_info_on_its_own_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    stock_logging = _stock_logging(monkeypatch)
    client = _server().client()
    guardrail = _embedded(client)
    second = _embedded(client)
    try:
        assert stock_logging.level == logging.INFO
        [handler] = stock_logging.handlers  # one, however many instances
        assert isinstance(handler, logging.StreamHandler)
        out = io.StringIO()
        handler.setStream(out)
        guardrail.heartbeat.summarize()
        assert "INFO collieai_guardrail: collieai guardrail summary {" in out.getvalue()
    finally:
        for built in (guardrail, second):
            await built.heartbeat.stop()
        await client.aclose()


async def test_a_handler_up_the_chain_takes_the_adapters_info_and_an_operators_level_is_kept(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stock_logging = _stock_logging(monkeypatch)
    out = io.StringIO()
    logging.getLogger().handlers.append(logging.StreamHandler(out))  # as JSON_LOGS gives the root one
    client = _server().client()
    guardrail = _embedded(client)
    try:
        assert stock_logging.level == logging.INFO and stock_logging.handlers == []
        guardrail.heartbeat.summarize()
        assert "collieai guardrail summary {" in out.getvalue()
        stock_logging.setLevel(logging.WARNING)  # the operator's choice
        await guardrail.heartbeat.stop()
        guardrail = _embedded(client)
        assert stock_logging.level == logging.WARNING
    finally:
        await guardrail.heartbeat.stop()
        await client.aclose()


async def test_a_handler_up_the_chain_that_drops_info_leaves_it_to_the_adapters_own_and_keeps_the_rest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A root handler at WARNING: `hasHandlers()` says yes, and the summary would show nowhere."""
    stock_logging = _stock_logging(monkeypatch)
    theirs = io.StringIO()
    root_handler = logging.StreamHandler(theirs)
    root_handler.setLevel(logging.WARNING)
    logging.getLogger().handlers.append(root_handler)
    client = _server().client()
    guardrail = _embedded(client)
    try:
        [own] = stock_logging.handlers
        assert isinstance(own, logging.StreamHandler)
        ours = io.StringIO()
        own.setStream(ours)
        guardrail.heartbeat.summarize()
        logging.getLogger("collieai_guardrail").warning("collieai guardrail: a warning")
        assert "collieai guardrail summary {" in ours.getvalue() and "a warning" not in ours.getvalue()
        assert "a warning" in theirs.getvalue() and "summary" not in theirs.getvalue()  # each line once
    finally:
        await guardrail.heartbeat.stop()
        await client.aclose()


async def test_one_handler_up_the_chain_that_takes_info_is_enough(monkeypatch: pytest.MonkeyPatch) -> None:
    stock_logging = _stock_logging(monkeypatch)
    warning_only = logging.StreamHandler(io.StringIO())
    warning_only.setLevel(logging.WARNING)
    logging.getLogger().handlers.extend([warning_only, logging.StreamHandler(io.StringIO())])
    client = _server().client()
    guardrail = _embedded(client)
    try:
        assert stock_logging.handlers == []
    finally:
        await guardrail.heartbeat.stop()
        await client.aclose()


async def test_a_logger_that_does_not_propagate_gets_its_own_handler_for_every_level(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The root's handler at WARNING never sees the adapter's records then, so its own takes them all."""
    stock_logging = _stock_logging(monkeypatch)
    root_handler = logging.StreamHandler(io.StringIO())
    root_handler.setLevel(logging.WARNING)
    logging.getLogger().handlers.append(root_handler)
    monkeypatch.setattr(stock_logging, "propagate", False)
    client = _server().client()
    guardrail = _embedded(client)
    try:
        [own] = stock_logging.handlers
        assert isinstance(own, logging.StreamHandler)
        ours = io.StringIO()
        own.setStream(ours)
        stock_logging.warning("collieai guardrail: a warning")
        guardrail.heartbeat.summarize()
        assert "a warning" in ours.getvalue() and "collieai guardrail summary {" in ours.getvalue()
    finally:
        await guardrail.heartbeat.stop()
        await client.aclose()


async def test_a_mode_without_post_call_or_with_default_off_warns(
    build: Build, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger="collieai_guardrail")
    build(_server(), mode="pre_call", default_on=False)
    messages = [r.getMessage() for r in caplog.records]
    assert any("mode has no post_call" in m for m in messages)
    assert any("default_on is false" in m for m in messages)
    assert not any("mode has no pre_call" in m for m in messages)


# ------------------------------------------------------------------ the hooks


async def test_pre_call_observes_records_in_litellms_log_and_changes_nothing(build: Build) -> None:
    server = _server()
    guardrail = build(server)
    data = _proxy_data()
    before = {
        key: copy.deepcopy(value) for key, value in data.items() if key not in ("metadata", "litellm_logging_obj")
    }
    result = await asyncio.wait_for(
        guardrail.async_pre_call_hook(user_api_key_dict=_user(), cache=None, data=data, call_type="acompletion"),
        timeout=5,
    )
    assert result is None
    assert {key: value for key, value in data.items() if key in before} == before
    assert len(server.requests) == 1 and server.bodies[0]["structured_messages"] == before["messages"]
    assert "request_headers" not in server.bodies[0]  # forwarding is off
    [record] = _records(data)
    assert record["guardrail_name"] == "collieai-monitor" and record["guardrail_provider"] == "collieai"
    assert record["guardrail_mode"] == GuardrailEventHooks.pre_call
    assert record["guardrail_status"] == "guardrail_flagged"  # the example would mask
    response = record["guardrail_response"]
    assert response["result"] == "completed" and response["effective"] == "allow" and response["would"] == "transform"
    assert response["operation_id"] == data["metadata"][OPERATION_ID_KEY]
    assert (
        server.bodies[0]["additional_provider_specific_params"]["collieai"]["operation_id"] == response["operation_id"]
    )
    assert server.bodies[0]["litellm_call_id"] == "call-1"
    assert SECRET_TEXT not in json.dumps(record, default=str) and "PELICAN" not in json.dumps(record, default=str)
    pre = guardrail.counters.snapshot()["pre_call"]
    assert (pre["hook_calls"], pre["eligible"], pre["attempted"], pre["completed"]) == (1, 1, 1, 1)


async def test_post_call_checks_litellms_response_under_the_same_operation_id(build: Build) -> None:
    server = _server()
    guardrail = build(server, forward_conversation_id=True)
    data = _proxy_data()
    await guardrail.async_pre_call_hook(user_api_key_dict=_user(), cache=None, data=data, call_type="acompletion")
    response = await _reply("the reply")
    response_before = response.model_dump()
    result = await asyncio.wait_for(
        guardrail.async_post_call_success_hook(data=data, user_api_key_dict=_user(), response=response), timeout=5
    )
    assert result is None and response.model_dump() == response_before
    assert server.bodies[1]["texts"] == ["the reply"] and server.bodies[1]["input_type"] == "response"
    operation_ids = {b["additional_provider_specific_params"]["collieai"]["operation_id"] for b in server.bodies}
    assert operation_ids == {data["metadata"][OPERATION_ID_KEY]}
    assert [b["request_headers"] for b in server.bodies] == [{"x-collieai-conversation-id": "c-9"}] * 2
    statuses = [(r["guardrail_mode"], r["guardrail_status"]) for r in _records(data)]
    assert statuses == [
        (GuardrailEventHooks.pre_call, "guardrail_flagged"),
        (GuardrailEventHooks.post_call, "guardrail_flagged"),
    ]
    assert guardrail.counters.snapshot()["post_call"]["completed"] == 1


async def test_pre_call_starts_a_new_logical_call_whatever_the_request_claims(build: Build) -> None:
    server = _server()
    guardrail = build(server)
    data = _proxy_data(metadata={OPERATION_ID_KEY: "client-chosen"})
    await guardrail.async_pre_call_hook(user_api_key_dict=_user(), cache=None, data=data, call_type="acompletion")
    assert data["metadata"][OPERATION_ID_KEY] != "client-chosen"


CARD_IN_HEX = b"4111111111111111".hex()  # a card number, written by the caller in the ID's very form


@pytest.mark.parametrize(
    "claimed",
    [
        {OPERATION_ID_KEY: "PRIVATE CUSTOMER TEXT PELICAN"},
        {OPERATION_ID_KEY: "0" * 31},
        {OPERATION_ID_KEY: "A" * 32},
        {OPERATION_ID_KEY: 7},
        {OPERATION_ID_KEY: CARD_IN_HEX},
        {OPERATION_ID_KEY: CARD_IN_HEX, OPERATION_TAG_KEY: "0" * 32},
        {OPERATION_ID_KEY: CARD_IN_HEX, OPERATION_TAG_KEY: "PELICAN é"},
        {OPERATION_ID_KEY: CARD_IN_HEX, OPERATION_TAG_KEY: 7},
        {OPERATION_ID_KEY: CARD_IN_HEX, OPERATION_TAG_KEY: CARD_IN_HEX},
    ],
)
async def test_post_call_takes_back_only_an_id_pre_call_issued(build: Build, claimed: dict[str, Any]) -> None:
    server = _server()
    guardrail = build(server)
    data = _proxy_data(metadata=dict(claimed))
    await guardrail.async_post_call_success_hook(data=data, user_api_key_dict=_user(), response=await _reply())
    [record] = _records(data)
    operation_id = record["guardrail_response"]["operation_id"]
    assert operation_id != claimed[OPERATION_ID_KEY] and OPERATION_ID_FORM.fullmatch(operation_id)
    assert data["metadata"][OPERATION_ID_KEY] == operation_id
    sent = json.dumps([record, server.bodies], default=str)
    assert "PELICAN" not in sent and CARD_IN_HEX not in sent


async def test_a_tag_pre_call_issued_for_one_call_vouches_for_no_other_id(build: Build) -> None:
    guardrail = build(_server())
    issued = _proxy_data()
    await guardrail.async_pre_call_hook(user_api_key_dict=_user(), cache=None, data=issued, call_type="acompletion")
    tag = issued["metadata"][OPERATION_TAG_KEY]
    data = _proxy_data(metadata={OPERATION_ID_KEY: CARD_IN_HEX, OPERATION_TAG_KEY: tag})
    await guardrail.async_post_call_success_hook(data=data, user_api_key_dict=_user(), response=await _reply())
    operation_id = _records(data)[0]["guardrail_response"]["operation_id"]
    assert operation_id not in (CARD_IN_HEX, issued["metadata"][OPERATION_ID_KEY])


@pytest.mark.parametrize(
    ("logging_call_type", "route", "expected"),
    [
        ("acompletion", None, "completed"),  # the logging object decides
        (None, ROUTE, "completed"),  # no logging object: the route, as LiteLLM's own guardrails resolve it
        (None, "/v1/responses", "outside"),
        (None, None, "outside"),  # neither: not known, so outside
        ("aresponses", ROUTE, "outside"),
    ],
)
async def test_post_call_learns_the_call_type_from_the_logging_object_or_else_the_route(
    build: Build, logging_call_type: str | None, route: str | None, expected: str
) -> None:
    server = _server()
    guardrail = build(server)
    data = chat_request()
    if logging_call_type is not None:
        data["litellm_logging_obj"] = SimpleNamespace(
            call_type=logging_call_type, litellm_call_id="call-1", litellm_trace_id="t-1"
        )
    await guardrail.async_post_call_success_hook(data=data, user_api_key_dict=_user(route), response=await _reply())
    [record] = _records(data)
    assert record["guardrail_response"]["result"] == expected
    if expected == "outside":
        assert record["guardrail_response"]["reason"] == "call_type" and record["guardrail_status"] == "not_run"
        assert server.requests == []


async def test_a_litellm_response_with_reasoning_is_outside_the_profile(build: Build) -> None:
    server = _server()
    guardrail = build(server)
    data = _proxy_data()
    response = ModelResponse(choices=[Choices(message=Message(content="safe", reasoning_content=SECRET_TEXT))])
    await guardrail.async_post_call_success_hook(data=data, user_api_key_dict=_user(), response=response)
    [record] = _records(data)
    assert (record["guardrail_status"], record["guardrail_response"]["reason"]) == ("not_run", "reasoning")
    assert server.requests == []


@pytest.mark.parametrize("data", [{}, {"messages": None}, {"messages": [{"role": "user", "content": 5}]}])
async def test_the_hooks_never_raise_on_a_request_of_any_shape(build: Build, data: dict[str, Any]) -> None:
    guardrail = build(_server())
    assert await guardrail.async_pre_call_hook(user_api_key_dict=None, cache=None, data=data, call_type=None) is None
    assert await guardrail.async_post_call_success_hook(data=data, user_api_key_dict=None, response=None) is None
    assert {r["guardrail_status"] for r in _records(data)} == {"not_run"}


async def test_a_fault_of_the_glue_is_logged_and_the_call_goes_on(
    build: Build, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger="collieai_guardrail")
    guardrail = build(_server())

    def broken(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError(SECRET_TEXT)

    monkeypatch.setattr(type(guardrail), "add_standard_logging_guardrail_information_to_request_data", broken)
    data = _proxy_data()
    assert (
        await guardrail.async_pre_call_hook(user_api_key_dict=_user(), cache=None, data=data, call_type="acompletion")
        is None
    )
    assert (
        await guardrail.async_post_call_success_hook(data=data, user_api_key_dict=_user(), response=await _reply())
        is None
    )
    messages = [r.getMessage() for r in caplog.records]
    assert messages == [
        "collieai guardrail pre_call: the glue failed (RuntimeError); the call goes on",
        "collieai guardrail post_call: the glue failed (RuntimeError); the call goes on",
    ]


async def test_the_calls_cancellation_goes_through_the_hook(build: Build) -> None:
    server = MockServer(never_answers)
    guardrail = build(server, deadline_ms=5000)
    task = asyncio.ensure_future(
        guardrail.async_pre_call_hook(
            user_api_key_dict=_user(), cache=None, data=_proxy_data(), call_type="acompletion"
        )
    )
    for _ in range(100):
        if server.requests:
            break
        await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=5)
    assert guardrail.counters.snapshot()["pre_call"]["bypass"] == {"cancelled": 1}


# ------------------------------------------------------------------ the count apart from the hooks


async def _until(condition: Callable[[], bool]) -> None:
    for _ in range(200):  # LiteLLM runs its async log callbacks as tasks after the call returns
        if condition():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("LiteLLM's log events did not arrive")


async def test_the_events_of_a_call_the_router_retried_count_once(build: Build) -> None:
    """As the real proxy's router makes them: each attempt is a `litellm.acompletion` with the request's
    metadata, so LiteLLM logs it with the ID pre-call issued. The caller's `x-litellm-call-id` becomes every attempt's
    `litellm_call_id` and also every other call's that sends it: it tells no call from another."""
    guardrail = build(_server())
    retried, other = _proxy_data(), _proxy_data()
    for data in (retried, other):
        await guardrail.async_pre_call_hook(user_api_key_dict=_user(), cache=None, data=data, call_type="acompletion")

    async def attempt(data: dict[str, Any], outcome: Any) -> None:
        await litellm.acompletion(
            model="gpt-4o-mini",
            messages=data["messages"],
            metadata=data["metadata"],
            litellm_call_id="caller-session-1",
            mock_response=outcome,
        )

    with pytest.raises(Exception):  # noqa: B017 — LiteLLM's mapped exception type is not the point
        await attempt(retried, Exception("down"))
    await attempt(retried, "the reply")
    await attempt(other, "the reply")

    def events() -> int:
        pre = guardrail.counters.snapshot()["pre_call"]
        return int(pre["logged_calls"]) + int(pre["logged_repeats"])  # type: ignore[call-overload]

    await _until(lambda: events() == 3)
    response = ModelResponse(choices=[Choices(message=Message(content="the reply"))])  # not a call: nothing logged
    for data in (retried, other):
        await guardrail.async_post_call_success_hook(data=data, user_api_key_dict=_user(), response=response)
    pre, post = guardrail.counters.snapshot()["pre_call"], guardrail.counters.snapshot()["post_call"]
    assert (pre["logged_calls"], pre["logged_eligible"], pre["logged_repeats"]) == (2, 2, 1)
    assert (post["logged_calls"], post["logged_eligible"]) == (2, 2)
    summary = guardrail.heartbeat.summarize()
    assert summary is not None
    assert [summary[phase]["logged_unseen"] for phase in ("pre_call", "post_call")] == [0, 0]  # type: ignore[index]


async def test_litellms_log_events_count_every_call_the_hooks_did_not_see(
    build: Build, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="collieai_guardrail")
    guardrail = build(_server())

    def logged() -> int:
        value = guardrail.counters.snapshot()["pre_call"]["logged_calls"]
        assert isinstance(value, int)
        return value

    await _reply()  # a call that never went through the proxy's hooks
    await _until(lambda: logged() == 1)
    with pytest.raises(Exception):  # noqa: B017 — LiteLLM's mapped exception type is not the point
        await litellm.acompletion(
            model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}], mock_response=Exception("down")
        )
    await _until(lambda: logged() == 2)
    stream = await litellm.acompletion(
        model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}], mock_response="a b", stream=True
    )
    async for _ in stream:
        pass
    await _until(lambda: logged() == 3)
    snapshot = guardrail.counters.snapshot()
    pre, post = snapshot["pre_call"], snapshot["post_call"]
    assert (pre["logged_calls"], pre["logged_eligible"], pre["logged_outside"]) == (3, 2, {"streaming": 1})
    assert (post["logged_calls"], post["logged_eligible"], post["logged_outside"]) == (2, 1, {"streaming": 1})
    assert pre["hook_calls"] == post["hook_calls"] == 0
    summary = guardrail.heartbeat.summarize()
    assert summary is not None
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert [w.split(":")[0] for w in warnings] == ["collieai guardrail pre_call", "collieai guardrail post_call"]
    assert all("not wired" in w for w in warnings)
