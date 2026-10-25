"""Fixtures of CollieAi's guardrail route for the adapter's tests: its requests, replies and errors, and a mock
server on httpx's MockTransport that answers by the route's error table — including a reply without JSON, one that
never comes, one that stops half way, and a dropped connection."""

from __future__ import annotations

import asyncio
import copy
import json
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx

API_BASE = "https://guardrail.example.test/v1/integrations/litellm"
URL = API_BASE + "/beta/litellm_basic_guardrail_api"
API_KEY = "clai_fake-key"  # not a real key
SECRET_TEXT = "my card is 4111 1111 1111 1111 and the code word is PELICAN"


def chat_request(**extra: Any) -> dict[str, Any]:
    """A proxy chat completion request as LiteLLM's pre-call hook gets it."""
    request: dict[str, Any] = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": SECRET_TEXT},
            {"role": "user", "content": [{"type": "text", "text": "and one more part"}]},
        ],
        "litellm_call_id": "call-1",
        "metadata": {"user_api_key_alias": "team-a"},
    }
    request.update(extra)
    return request


def chat_response(content: Any = "a reply with " + SECRET_TEXT, **message: Any) -> dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": "gpt-4o-mini",
        "choices": [
            {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content, **message}}
        ],
    }


def check_reply(phase: str = "pre_call", items: int = 3, *, found: bool = True) -> dict[str, Any]:
    """The route's example reply, as the server serializes it; without `found`, the same
    check that found nothing."""
    if not found:
        reply = check_reply(phase, items)
        reply["collieai"]["would"].update(outcome="allow", masked_items=[], would_mask=False)
        del reply["collieai"]["items"]
        return reply
    return {
        "action": "NONE",
        "collieai": {
            "version": 1,
            "evaluation_id": "9f1c2d3e-0000-4000-8000-000000000001",
            "phase": phase,
            "terminal": "completed",
            "profile": "stage1a-narrow-v1",
            "policy_fingerprint": "ps:abc123",
            "mode": "monitor",
            "coverage": {
                "state": "clean",
                "items_judged": items,
                "items_total": items,
                "rules_run": [5] * items,
                "degraded_rules": [],
            },
            "would": {
                "outcome": "transform",
                "complete": True,
                "approximate": False,
                "masked_items": [2],
                "would_block": False,
                "would_mask": True,
            },
            "items": [
                {
                    "index": 2,
                    "source": "structured_messages[2].content",
                    "outcome": "transform",
                    "findings": [
                        {
                            "rule_id": "7d1e",
                            "rule_type": "aho_corasick",
                            "action": "mask",
                            "mode": "monitor",
                            "kind": "detection",
                        }
                    ],
                }
            ],
            "timings_ms": {"normalize": 0.4, "pin": 6.1, "check": 0.2, "judge": 41.0, "total": 48.3},
        },
    }


def legacy_reply(phase: str = "pre_call", items: int = 3) -> dict[str, Any]:
    """The example as a server before finding kinds (2026.10.23) serializes it: no finding names its `kind`."""
    reply = check_reply(phase, items)
    for item in reply["collieai"].get("items", []):
        for finding in item["findings"]:
            del finding["kind"]
    return reply


def error_body(error_type: str, code: str | None = None) -> dict[str, Any]:
    return {"error": {"message": "fixed text", "type": error_type, "code": code}}


#: The route's error table: (status, body, the adapter's bypass reason).
ERROR_TABLE: list[tuple[int, dict[str, Any], str]] = [
    (404, {"error": {"message": "Not Found", "type": "not_found_error"}}, "not_found"),
    (401, {"error": {"message": "Unauthorized", "type": "authentication_error"}}, "auth"),
    (403, {"error": {"message": "Forbidden", "type": "forbidden_error"}}, "auth"),
    (
        429,
        {"error": {"message": "Rate limit exceeded", "type": "rate_limit_error", "code": "rate_limited"}},
        "rate_limited",
    ),
    (413, error_body("limit", "body_too_large"), "limit"),
    (422, error_body("malformed", "malformed"), "protocol"),
    (422, error_body("shape", "tool_calls"), "shape"),
    (422, error_body("limit", "total_chars"), "limit"),
    (422, error_body("capability", "pin_unsupported"), "capability"),
    (422, error_body("protocol", "unsupported_contract_version"), "protocol"),
    (503, error_body("busy", "busy"), "busy"),
    (503, error_body("pin", "pin_unavailable"), "unavailable"),
    (504, error_body("deadline", "deadline_expired"), "deadline"),
    (500, error_body("pin", "config_corrupt"), "server_error"),
    (500, error_body("execution", "execution_error"), "server_error"),
    (500, error_body("internal", "invariant_violated"), "server_error"),
]


class MockServer:
    """Answers every POST with `respond(request)` and keeps each request it got."""

    def __init__(self, respond: Callable[[httpx.Request], Any]) -> None:
        self.respond = respond
        self.requests: list[httpx.Request] = []
        self.bodies: list[dict[str, Any]] = []

    async def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        self.bodies.append(json.loads(request.content))
        answer = self.respond(request)
        if asyncio.iscoroutine(answer):
            answer = await answer
        assert isinstance(answer, httpx.Response)
        return answer

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handle), follow_redirects=False)


class _Body(httpx.AsyncByteStream):
    """A reply body that arrives as from a socket. `httpx.Response(content=...)` reads its body when it is built,
    which no network reply does."""

    def __init__(self, content: bytes) -> None:
        self.content = content

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self.content


def reply(status: int, content: bytes = b"", headers: dict[str, str] | None = None) -> httpx.Response:
    return httpx.Response(status, headers=headers, stream=_Body(content))


def json_reply(status: int, body: Any) -> Callable[[httpx.Request], httpx.Response]:
    content = json.dumps(copy.deepcopy(body)).encode()
    return lambda request: reply(status, content, {"content-type": "application/json"})


def raw_reply(status: int, content: bytes) -> Callable[[httpx.Request], httpx.Response]:
    return lambda request: reply(status, content)


def never_answers(request: httpx.Request) -> Any:
    async def hang() -> httpx.Response:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    return hang()


def refuses_connection(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("connection refused", request=request)


class _Stream(httpx.AsyncByteStream):
    def __init__(self, first: bytes, then: str) -> None:
        self.first, self.then = first, then

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self.first
        if self.then == "hang":
            await asyncio.Event().wait()
        raise httpx.RemoteProtocolError("peer closed connection without sending complete message body")


def stops_half_way(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, stream=_Stream(b'{"action": "NO', "drop"))


def body_never_ends(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, stream=_Stream(b'{"action": "NO', "hang"))
