"""The request to CollieAi's guardrail route.

A `GenericGuardrailAPIRequest`: pre-call sends the request's messages as they are in `structured_messages` and
`texts: null`; post-call sends the response's one text in `texts`. The `collieai` extension is always sent — it asks
for the diagnostics the strict parse needs and gives the server its budget. `request_data` is `{}`: the schema
requires it and the server never reads it. Nothing of the user's request beyond the messages and the model's name
goes out; the conversation ID header only when the facts carry one, which the observer lets them do only when the
operator turned its forwarding on.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass

CONTRACT_VERSION = 1
CONVERSATION_HEADER = "x-collieai-conversation-id"


@dataclass(frozen=True)
class CallFacts:
    """A logical call's correlation and labels, the same for its pre and post phases; none of them is authority.
    `operation_id` is the adapter's own, one per logical call."""

    operation_id: str
    litellm_call_id: str | None = None
    litellm_trace_id: str | None = None
    model: str | None = None
    conversation_id: str | None = None


@dataclass(frozen=True)
class Sender:
    """What every request of one adapter instance says about the adapter."""

    adapter_version: str
    litellm_version: str | None
    budget_ms: int
    #: `block` asks the server for enforce with blocking only; a Monitor request does not name it.
    enforcement: str = "monitor"


def pre_call_payload(request: Mapping[str, object], facts: CallFacts, sender: Sender) -> dict[str, object]:
    return {
        "input_type": "request",
        "structured_messages": request.get("messages"),
        "texts": None,
        **_common(facts, sender, stream=bool(request.get("stream")), n=_n(request)),
    }


def post_call_payload(request: Mapping[str, object], text: str, facts: CallFacts, sender: Sender) -> dict[str, object]:
    return {
        "input_type": "response",
        "texts": [text],
        **_common(facts, sender, stream=bool(request.get("stream")), n=_n(request)),
    }


def encode(payload: Mapping[str, object]) -> bytes:
    """Compact UTF-8 JSON. Raises TypeError or ValueError on a value JSON cannot carry."""
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _n(request: Mapping[str, object]) -> int:
    n = request.get("n")
    return n if type(n) is int else 1


def _common(facts: CallFacts, sender: Sender, *, stream: bool, n: int) -> dict[str, object]:
    collieai: dict[str, object] = {
        "contract_version": CONTRACT_VERSION,
        "adapter": "native",
        "adapter_version": sender.adapter_version,
        "operation_id": facts.operation_id,
        "source_endpoint": "chat_completions",
        "budget_ms": sender.budget_ms,
        "stream": stream,
        "n": n,
    }
    if sender.enforcement == "block":
        collieai["enforcement"] = "block"
    common: dict[str, object] = {
        "request_data": {},
        "litellm_call_id": facts.litellm_call_id,
        "litellm_trace_id": facts.litellm_trace_id,
        "model": facts.model,
        "litellm_version": sender.litellm_version,
        "additional_provider_specific_params": {"collieai": collieai},
    }
    if facts.conversation_id:
        common["request_headers"] = {CONVERSATION_HEADER: facts.conversation_id}
    return common
