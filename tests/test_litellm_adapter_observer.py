"""One phase end to end against a mock server: one POST, the deadline, bypass by reason, the call unchanged, the
counters and the logs."""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import socket
import time
from collections.abc import Callable
from typing import Any

import httpx
import pytest

import collieai_guardrail.observer as observer_module
from collieai_guardrail.config import AdapterConfig
from collieai_guardrail.counters import Counters
from collieai_guardrail.observer import Observer, Outcome
from collieai_guardrail.payload import CallFacts, encode
from collieai_guardrail.transport import Failure, Reply, new_client, post_once
from collieai_guardrail.verdict import Check, parse_check
from litellm_contract import (
    API_BASE,
    API_KEY,
    ERROR_TABLE,
    SECRET_TEXT,
    URL,
    MockServer,
    body_never_ends,
    chat_request,
    chat_response,
    check_reply,
    json_reply,
    never_answers,
    raw_reply,
    refuses_connection,
    reply,
    stops_half_way,
)

DEADLINE_MS = 300
#: What a phase may take beyond its deadline: the event loop's wake-up and the bookkeeping after it.
SLACK_S = 0.25
FACTS = CallFacts(operation_id="op-1", litellm_call_id="call-1", model="gpt-4o-mini")


def _observer(server: MockServer, **config: Any) -> Observer:
    settings: dict[str, Any] = {
        "api_base": API_BASE,
        "api_key": API_KEY,
        "deadline_ms": DEADLINE_MS,
        "network_margin_ms": 100,
        **config,
    }
    return Observer(AdapterConfig(**settings), litellm_version="1.102.4", client=server.client())


async def _pre(observer: Observer, request: dict[str, Any] | None = None, call_type: str = "acompletion") -> Outcome:
    # the test's own bound: a regression that hangs fails here instead of hanging the suite
    return await asyncio.wait_for(
        observer.observe_request(request if request is not None else chat_request(), call_type=call_type, facts=FACTS),
        timeout=5,
    )


async def _post(
    observer: Observer,
    response: Any = None,
    request: dict[str, Any] | None = None,
    call_type: str | None = "acompletion",
) -> Outcome:
    return await asyncio.wait_for(
        observer.observe_response(
            request if request is not None else chat_request(),
            response if response is not None else chat_response(),
            call_type=call_type,
            facts=FACTS,
        ),
        timeout=5,
    )


# ------------------------------------------------------------------ a check


async def test_a_check_is_one_post_with_the_key_and_the_extension_and_counts_as_completed() -> None:
    server = MockServer(json_reply(200, check_reply("pre_call", found=False)))
    observer = _observer(server)
    outcome = await _pre(observer)
    assert outcome.result == "completed" and outcome.guardrail_status == "success"
    assert outcome.check is not None and outcome.check.evaluation_id.startswith("9f1c")
    assert len(server.requests) == 1
    sent = server.requests[0]
    assert str(sent.url) == URL and sent.method == "POST"
    assert sent.headers["authorization"] == f"Bearer {API_KEY}"
    assert sent.headers["accept-encoding"] == "identity"
    assert server.bodies[0]["structured_messages"] == chat_request()["messages"]
    assert server.bodies[0]["additional_provider_specific_params"]["collieai"]["budget_ms"] == DEADLINE_MS - 100
    pre = observer.counters.snapshot()["pre_call"]
    assert (pre["hook_calls"], pre["eligible"], pre["attempted"], pre["completed"]) == (1, 1, 1, 1)
    assert pre["bypass"] == {} and pre["outside"] == {}


async def test_post_call_sends_the_responses_text_and_parses_its_own_phase() -> None:
    server = MockServer(json_reply(200, check_reply("post_call", items=1)))
    observer = _observer(server)
    outcome = await _post(observer, chat_response("the reply"))
    assert outcome.result == "completed"
    assert server.bodies[0]["texts"] == ["the reply"] and server.bodies[0]["input_type"] == "response"


async def test_an_empty_response_is_sent_and_checked() -> None:
    server = MockServer(json_reply(200, check_reply("post_call", items=1, found=False)))
    outcome = await _post(_observer(server), chat_response(""))
    assert outcome.result == "completed"
    assert server.bodies[0]["texts"] == [""]


@pytest.mark.parametrize("call_type", ["aresponses", None])
async def test_post_call_of_a_call_that_is_no_chat_completion_sends_nothing(call_type: str | None) -> None:
    """A chat-shaped request and response do not make a call a Chat Completion: post-call goes by the call's type,
    as pre-call does, and a type it does not know is outside too."""
    server = MockServer(json_reply(200, check_reply("post_call", items=1)))
    observer = _observer(server)
    assert (await _pre(observer, call_type="aresponses")).reason == "call_type"
    outcome = await _post(observer, call_type=call_type)
    assert (outcome.result, outcome.reason) == ("outside", "call_type")
    assert server.requests == []
    assert observer.counters.snapshot()["post_call"]["outside"] == {"call_type": 1}


async def test_a_check_that_found_something_is_flagged_and_still_changes_nothing() -> None:
    request, response = chat_request(), chat_response()
    before = copy.deepcopy((request, response))
    found = await _pre(_observer(MockServer(json_reply(200, check_reply("pre_call")))), request)
    assert (found.result, found.guardrail_status) == ("completed", "guardrail_flagged")
    assert found.diagnostics()["effective"] == "allow" and found.diagnostics()["would"] == "transform"
    clean = await _post(_observer(MockServer(json_reply(200, check_reply("post_call", 1, found=False)))), response)
    assert (clean.result, clean.guardrail_status) == ("completed", "success")
    assert (request, response) == before


async def test_a_reply_for_the_other_phase_is_a_protocol_bypass() -> None:
    server = MockServer(json_reply(200, check_reply("pre_call")))
    outcome = await _post(_observer(server))
    assert (outcome.result, outcome.reason, outcome.detail) == ("bypass", "protocol", "phase")


def test_the_diagnostics_keep_would_effective_and_degraded_apart() -> None:
    check = parse_check(json.dumps(check_reply()).encode(), phase="pre_call")
    assert isinstance(check, Check)
    diagnostics = Outcome("pre_call", "completed", check=check).diagnostics()
    assert diagnostics["effective"] == "allow"
    assert diagnostics["would"] == "transform"
    assert diagnostics["degraded"] is False
    assert diagnostics["collieai"] == check_reply()["collieai"]
    bypass = Outcome("pre_call", "bypass", reason="busy", http_status=503).diagnostics()
    assert bypass == {
        "adapter": "collieai-native",
        "phase": "pre_call",
        "result": "bypass",
        "effective": "allow",
        "reason": "busy",
        "http_status": 503,
    }
    assert Outcome("pre_call", "bypass", reason="busy").guardrail_status == "guardrail_failed_to_respond"
    assert Outcome("pre_call", "outside", reason="streaming").guardrail_status == "not_run"


# ------------------------------------------------------------------ bypass by reason, one POST each


@pytest.mark.parametrize(("status", "body", "reason"), ERROR_TABLE)
async def test_each_error_of_the_table_is_a_bypass_with_its_reason_after_one_post(
    status: int, body: Any, reason: str
) -> None:
    server = MockServer(json_reply(status, body))
    observer = _observer(server)
    outcome = await _pre(observer)
    assert (outcome.result, outcome.reason, outcome.http_status) == ("bypass", reason, status)
    assert len(server.requests) == 1
    assert observer.counters.snapshot()["pre_call"]["bypass"] == {reason: 1}


@pytest.mark.parametrize(
    ("respond", "reason", "detail"),
    [
        (raw_reply(502, b"<html>Bad Gateway</html>"), "http_5xx", None),
        (raw_reply(503, b""), "http_5xx", None),
        (raw_reply(200, b"<html>ok</html>"), "protocol", "not_json"),
        (raw_reply(200, b"[]"), "protocol", "not_object"),
        (raw_reply(204, b""), "protocol", "status"),
        (lambda request: reply(302, headers={"location": URL}), "http_3xx", None),
        (refuses_connection, "transport", "ConnectError"),
        (stops_half_way, "transport", "RemoteProtocolError"),
    ],
)
async def test_a_reply_that_is_no_verdict_is_a_bypass_after_exactly_one_post(
    respond: Callable[[httpx.Request], Any], reason: str, detail: str | None
) -> None:
    server = MockServer(respond)
    outcome = await _pre(_observer(server))
    assert (outcome.result, outcome.reason) == ("bypass", reason)
    if detail is not None:
        assert outcome.detail == detail
    assert len(server.requests) == 1  # no retry: not on 503, not on a refused or dropped connection, no redirect


@pytest.mark.parametrize("respond", [never_answers, body_never_ends])
async def test_the_deadline_holds_against_a_server_that_never_answers(respond: Callable[[httpx.Request], Any]) -> None:
    server = MockServer(respond)
    observer = _observer(server)
    started = time.monotonic()
    outcome = await _pre(observer)
    elapsed = time.monotonic() - started
    assert (outcome.result, outcome.reason) == ("bypass", "timeout")
    assert DEADLINE_MS / 1000 - 0.05 <= elapsed <= DEADLINE_MS / 1000 + SLACK_S
    assert len(server.requests) == 1


@pytest.mark.parametrize(
    ("status", "content"),
    [
        (200, json.dumps(check_reply()).encode()),
        (200, b"<html>ok</html>"),  # would be protocol/not_json in time
        (503, json.dumps({"error": {"type": "busy"}}).encode()),  # would be busy in time
        (401, b""),  # would be http_4xx in time
    ],
)
async def test_a_reply_read_after_the_deadline_is_a_timeout_whatever_it_says(status: int, content: bytes) -> None:
    def slow_but_blocking(request: httpx.Request) -> httpx.Response:
        time.sleep(DEADLINE_MS / 1000 + 0.05)  # holds the loop, so no timer can interrupt the exchange
        return reply(status, content)

    server = MockServer(slow_but_blocking)
    observer = _observer(server)
    outcome = await _pre(observer)
    assert (outcome.result, outcome.reason, outcome.detail) == ("bypass", "timeout", "late")
    assert outcome.http_status == status and outcome.check is None
    pre = observer.counters.snapshot()["pre_call"]
    assert pre["completed"] == 0 and pre["bypass"] == {"timeout": 1}


async def test_a_parse_that_ends_after_the_deadline_is_a_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """The deadline covers the parse too, on every path of the reply — a refused 200 as well as a check."""

    def slow_parse(body: bytes, *, phase: str) -> Any:
        time.sleep(DEADLINE_MS / 1000 + 0.05)
        return parse_check(body, phase=phase)

    monkeypatch.setattr(observer_module, "parse_check", slow_parse)
    outcome = await _pre(_observer(MockServer(raw_reply(200, b"not json"))))
    assert (outcome.result, outcome.reason, outcome.detail) == ("bypass", "timeout", "late")


async def test_a_request_over_the_body_limit_is_not_sent() -> None:
    server = MockServer(json_reply(200, check_reply()))
    observer = _observer(server, max_body_bytes=200)
    outcome = await _pre(observer)
    assert (outcome.result, outcome.reason, outcome.detail) == ("bypass", "limit", "body_over_limit")
    assert server.requests == []
    pre = observer.counters.snapshot()["pre_call"]
    assert (pre["eligible"], pre["attempted"]) == (1, 0)


async def test_an_error_inside_the_adapter_never_escapes_and_is_counted() -> None:
    server = MockServer(json_reply(200, check_reply()))
    observer = _observer(server)
    request = chat_request()
    request["messages"][0]["extra"] = {1, 2}  # nothing JSON can carry
    outcome = await _pre(observer, request)
    assert (outcome.result, outcome.reason, outcome.detail) == ("bypass", "adapter_error", "TypeError")
    assert server.requests == []


async def test_a_reply_over_the_size_limit_is_not_read_in_full() -> None:
    async def big(request: httpx.Request) -> httpx.Response:
        return reply(200, b" " * 4096)

    async with httpx.AsyncClient(transport=httpx.MockTransport(big)) as client:
        loop = asyncio.get_running_loop()
        sent = await post_once(client, url=URL, headers={}, body=b"{}", deadline=loop.time() + 5, max_reply_bytes=1024)
    assert sent == Failure("protocol", "reply_too_large")


async def test_no_time_left_sends_nothing_and_is_no_attempt() -> None:
    server = MockServer(json_reply(200, check_reply()))
    attempts: list[None] = []
    async with server.client() as client:
        loop = asyncio.get_running_loop()
        late = await post_once(
            client,
            url=URL,
            headers={},
            body=b"{}",
            deadline=loop.time() - 0.001,
            max_reply_bytes=1024,
            on_attempt=lambda: attempts.append(None),
        )
        assert late == Failure("timeout", "no_time_left")
        assert server.requests == [] and attempts == []
        sent = await post_once(
            client,
            url=URL,
            headers={},
            body=b"{}",
            deadline=loop.time() + 5,
            max_reply_bytes=1024 * 1024,
            on_attempt=lambda: attempts.append(None),
        )
    assert isinstance(sent, Reply) and len(server.requests) == 1 and attempts == [None]


async def test_a_deadline_gone_before_the_post_started_counts_no_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    """`attempted` is the transport's word that the POST started, not the adapter's intent to send."""

    def slow_encode(payload: Any) -> bytes:
        time.sleep(DEADLINE_MS / 1000 + 0.05)  # the request built, the deadline gone
        return encode(payload)

    monkeypatch.setattr(observer_module, "encode", slow_encode)
    server = MockServer(json_reply(200, check_reply()))
    observer = _observer(server)
    outcome = await _pre(observer)
    assert (outcome.result, outcome.reason, outcome.detail) == ("bypass", "timeout", "no_time_left")
    assert server.requests == []
    pre = observer.counters.snapshot()["pre_call"]
    assert (pre["eligible"], pre["attempted"], pre["completed"]) == (1, 0, 0)
    assert pre["bypass"] == {"timeout": 1}


async def test_a_refused_connection_is_an_attempt() -> None:
    observer = _observer(MockServer(refuses_connection))
    await _pre(observer)
    assert observer.counters.snapshot()["pre_call"]["attempted"] == 1


async def test_the_adapters_own_client_follows_no_redirect() -> None:
    server = MockServer(lambda request: reply(307, headers={"location": URL + "/elsewhere"}))
    config = AdapterConfig(api_base=API_BASE, api_key=API_KEY, deadline_ms=DEADLINE_MS, network_margin_ms=100)
    observer = Observer(config, client=new_client(transport=httpx.MockTransport(server.handle)))
    try:
        outcome = await _pre(observer)
    finally:
        await observer.aclose()
    assert (outcome.result, outcome.reason, outcome.http_status) == ("bypass", "http_3xx", 307)
    assert len(server.requests) == 1


async def test_the_adapters_own_client_does_not_retry_a_refused_connection() -> None:
    """httpx's transport retries a failed connect with a growing pause when built with `retries`; the adapter's is
    built with none, so a refused connection is a bypass at once."""
    with socket.socket() as probe:  # a port nothing listens on
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    config = AdapterConfig(
        api_base=f"http://127.0.0.1:{port}/v1/integrations/litellm",
        api_key=API_KEY,
        deadline_ms=2000,
        network_margin_ms=500,
    )
    observer = Observer(config, client=new_client())
    started = time.monotonic()
    try:
        outcome = await _pre(observer)
    finally:
        await observer.aclose()
    assert (outcome.result, outcome.reason, outcome.detail) == ("bypass", "transport", "ConnectError")
    assert time.monotonic() - started < 0.4


async def test_the_real_client_does_not_repeat_a_post_whose_connection_dropped() -> None:
    """Stock generic's shared client posts once more after RemoteProtocolError or ConnectError; the
    adapter's own client, over a real socket, does not."""
    connections = 0

    async def drop(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        nonlocal connections
        connections += 1
        await reader.read(65536)
        writer.close()

    tcp = await asyncio.start_server(drop, "127.0.0.1", 0)
    port = tcp.sockets[0].getsockname()[1]
    try:
        config = AdapterConfig(
            api_base=f"http://127.0.0.1:{port}/v1/integrations/litellm",
            api_key=API_KEY,
            deadline_ms=2000,
            network_margin_ms=500,
        )
        observer = Observer(config, client=new_client())
        try:
            outcome = await _pre(observer)
        finally:
            await observer.aclose()
    finally:
        tcp.close()
        await tcp.wait_closed()
    assert (outcome.result, outcome.reason) == ("bypass", "transport")
    assert connections == 1


# ------------------------------------------------------------------ the call unchanged


@pytest.mark.parametrize(
    "respond",
    [
        json_reply(200, check_reply()),
        json_reply(503, {"error": {"type": "busy"}}),
        refuses_connection,
        never_answers,
        raw_reply(200, b"not json"),
    ],
)
async def test_the_request_and_the_response_are_never_changed(respond: Callable[[httpx.Request], Any]) -> None:
    server = MockServer(respond)
    observer = _observer(server)
    request, response = chat_request(), chat_response()
    request_before, response_before = copy.deepcopy(request), copy.deepcopy(response)
    await _pre(observer, request)
    await _post(observer, response, request)
    assert request == request_before
    assert response == response_before


async def test_a_call_outside_the_profile_sends_nothing_and_is_counted_by_reason() -> None:
    server = MockServer(json_reply(200, check_reply()))
    observer = _observer(server)
    streaming = chat_request(stream=True)
    assert (await _pre(observer, streaming)).reason == "streaming"
    assert (await _pre(observer, call_type="aresponses")).reason == "call_type"
    assert (await _post(observer, request=streaming)).reason == "streaming"
    assert (await _post(observer, chat_response(None, tool_calls=[{"id": "t"}]))).reason == "tool_calls"
    assert (await _post(observer, chat_response("safe", reasoning_content=SECRET_TEXT))).reason == "reasoning"
    assert server.requests == []
    snapshot = observer.counters.snapshot()
    assert snapshot["pre_call"]["outside"] == {"call_type": 1, "streaming": 1}
    assert snapshot["post_call"]["outside"] == {"streaming": 1, "tool_calls": 1, "reasoning": 1}
    assert snapshot["pre_call"]["eligible"] == snapshot["post_call"]["eligible"] == 0


async def test_every_eligible_call_ends_as_exactly_one_completed_or_bypass() -> None:
    replies = iter([check_reply(), {"error": {"type": "busy"}}, check_reply(), None])

    def respond(request: httpx.Request) -> httpx.Response:
        answer = next(replies)
        if answer is None:
            raise httpx.ConnectError("refused", request=request)
        return json_reply(503 if "error" in answer else 200, answer)(request)

    observer = _observer(MockServer(respond))
    for _ in range(4):
        await _pre(observer)
    await _pre(observer, chat_request(stream=True))
    pre = observer.counters.snapshot()["pre_call"]
    assert pre["hook_calls"] == 5 and pre["eligible"] == 4
    bypassed = pre["bypass"]
    assert isinstance(bypassed, dict)
    assert pre["completed"] == 2 and sum(bypassed.values()) == 2
    assert pre["bypass"] == {"busy": 1, "transport": 1}


async def test_a_synthetic_observation_leaves_the_calls_counts_alone() -> None:
    observer = _observer(MockServer(json_reply(200, check_reply())))
    outcome = await asyncio.wait_for(
        observer.observe_request(chat_request(), call_type="acompletion", facts=FACTS, count=False), timeout=5
    )
    assert outcome.result == "completed"
    assert observer.counters.snapshot() == Counters().snapshot()


@pytest.mark.parametrize("phase", ["pre_call", "post_call"])
async def test_the_callers_cancellation_goes_on_and_the_eligible_call_still_ends_once(phase: str) -> None:
    server = MockServer(never_answers)
    observer = _observer(server, deadline_ms=5000, network_margin_ms=100)
    observe = (
        observer.observe_request(chat_request(), call_type="acompletion", facts=FACTS)
        if phase == "pre_call"
        else observer.observe_response(chat_request(), chat_response(), call_type="acompletion", facts=FACTS)
    )
    task = asyncio.ensure_future(observe)
    for _ in range(100):  # until the POST is out
        if server.requests:
            break
        await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=5)
    counts = observer.counters.snapshot()[phase]
    assert (counts["eligible"], counts["attempted"], counts["completed"]) == (1, 1, 0)
    assert counts["bypass"] == {"cancelled": 1}


# ------------------------------------------------------------------ the conversation ID


@pytest.mark.parametrize("forward", [False, True])
async def test_the_conversation_id_leaves_only_when_the_operator_turned_its_forwarding_on(forward: bool) -> None:
    server = MockServer(json_reply(200, check_reply("pre_call")))
    observer = _observer(server, forward_conversation_id=forward)
    facts = CallFacts(operation_id="op-1", conversation_id="conv-9")
    await asyncio.wait_for(observer.observe_request(chat_request(), call_type="acompletion", facts=facts), timeout=5)
    server.respond = json_reply(200, check_reply("post_call", items=1))
    await asyncio.wait_for(
        observer.observe_response(chat_request(), chat_response(), call_type="acompletion", facts=facts), timeout=5
    )
    sent = [body.get("request_headers") for body in server.bodies]
    assert sent == ([{"x-collieai-conversation-id": "conv-9"}] * 2 if forward else [None, None])


# ------------------------------------------------------------------ the logs


async def test_logs_carry_no_text_and_no_key_and_a_bypass_warns_once_a_minute(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG, logger="collieai_guardrail")
    observer = _observer(MockServer(json_reply(503, {"error": {"type": "busy"}})))
    for _ in range(3):
        await _pre(observer)
    await _post(observer)
    await _pre(_observer(MockServer(json_reply(200, check_reply()))))
    await _pre(observer, chat_request(stream=True))
    text = "\n".join(record.getMessage() for record in caplog.records)
    for secret in (SECRET_TEXT, "PELICAN", "4111", API_KEY, "call-1", "op-1"):
        assert secret not in text
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert [w.getMessage().split(":")[0] for w in warnings] == [
        "collieai guardrail pre_call",
        "collieai guardrail post_call",
    ]
