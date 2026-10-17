"""The adapter's settings, which calls are in the supported profile, and the request it sends."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace
from typing import Any

import pytest

from collieai_guardrail.config import AdapterConfig, ConfigError
from collieai_guardrail.eligibility import OUTSIDE_REASONS, request_outside_reason, response_text
from collieai_guardrail.payload import CallFacts, Sender, encode, post_call_payload, pre_call_payload
from litellm_contract import API_BASE, API_KEY, URL, chat_request, chat_response

# ------------------------------------------------------------------ settings


def test_settings_give_the_route_and_the_servers_share_of_the_deadline() -> None:
    config = AdapterConfig(api_base=API_BASE, api_key=API_KEY, deadline_ms=2500, network_margin_ms=500)
    assert config.url == URL
    assert AdapterConfig(api_base=API_BASE + "/", api_key=API_KEY).url == URL
    assert config.budget_ms == 2000


@pytest.mark.parametrize(
    "change",
    [
        {"api_base": "ftp://guardrail.example.test"},
        {"api_base": "guardrail.example.test/v1"},
        {"api_base": "https://user:pw@guardrail.example.test/v1"},
        {"api_base": "https://guardrail.example.test/v1?x=1"},
        {"api_base": "https://["},  # urlsplit's own ValueError, which LiteLLM's loader would swallow
        {"api_base": "https://guardrail.example.test:99999/v1"},
        {"api_base": "https://guardrail.example.test:port/v1"},
        {"api_base": "https://guardrail.example.test:0/v1"},
        {"api_key": ""},
        {"api_key": " " + API_KEY},
        {"api_key": API_KEY[:5] + "\n" + API_KEY[5:]},
        {"deadline_ms": 0},
        {"deadline_ms": 60_001},
        {"deadline_ms": 2500.0},
        {"deadline_ms": True},
        {"network_margin_ms": -1},
        {"network_margin_ms": 2500},  # the server would get no budget
        {"max_body_bytes": 0},
        {"forward_conversation_id": "yes"},
    ],
)
def test_settings_refuse_what_the_adapter_cannot_run_with_without_echoing_the_key(change: dict[str, Any]) -> None:
    values: dict[str, Any] = {"api_base": API_BASE, "api_key": API_KEY, "deadline_ms": 2500, **change}
    with pytest.raises(ConfigError) as refused:
        AdapterConfig(**values)
    assert API_KEY not in str(refused.value) and API_KEY[5:] not in str(refused.value)


@pytest.mark.parametrize(
    "api_base",
    [
        "http://guardrail.example.test/v1/integrations/litellm",
        "http://app.collieai.io/v1/integrations/litellm",
        "http://10.0.0.5:8000/v1/integrations/litellm",  # a private address is still the network
        "http://localhost.example.test/v1",  # only localhost itself is trusted to stay on the machine
        "http://128.0.0.1/v1",
    ],
)
def test_http_is_refused_off_this_machine(api_base: str) -> None:
    """The key and the caller's text would cross the network in clear."""
    with pytest.raises(ConfigError, match="must use https"):
        AdapterConfig(api_base=api_base, api_key=API_KEY)


@pytest.mark.parametrize(
    "api_base",
    [
        "https://guardrail.example.test/v1/integrations/litellm",
        "http://127.0.0.1:8411/v1/integrations/litellm",
        "http://127.1.2.3/v1",
        "http://[::1]:8411/v1",
        "http://localhost:8411/v1",
        "http://LOCALHOST/v1",
    ],
)
def test_https_or_a_loopback_stand_is_accepted(api_base: str) -> None:
    assert AdapterConfig(api_base=api_base, api_key=API_KEY).url.endswith("/beta/litellm_basic_guardrail_api")


def test_the_key_is_not_in_the_settings_repr() -> None:
    assert API_KEY not in repr(AdapterConfig(api_base=API_BASE, api_key=API_KEY))


# ------------------------------------------------------------------ which calls are in the supported profile


def test_a_text_chat_completion_is_in_the_profile_and_reading_it_changes_nothing() -> None:
    request = chat_request(stream=False, n=1)
    before = copy.deepcopy(request)
    assert request_outside_reason(request, "acompletion") is None
    assert request_outside_reason(request, "completion") is None
    assert request == before


def test_empty_texts_are_texts() -> None:
    """The server judges an empty string as an item, so the adapter sends it; only a missing or null content is
    outside the supported profile."""
    empty = chat_request(
        messages=[{"role": "user", "content": ""}, {"role": "user", "content": [{"type": "text", "text": ""}]}]
    )
    assert request_outside_reason(empty, "acompletion") is None
    assert response_text(chat_response("")) == ("", None)


@pytest.mark.parametrize(
    ("change", "call_type", "reason"),
    [
        ({}, "aresponses", "call_type"),
        ({}, "aembedding", "call_type"),
        ({}, None, "call_type"),  # a type not known: a chat-shaped request does not make the call a Chat Completion
        ({}, "anthropic_messages", "call_type"),
        ({"stream": True}, "acompletion", "streaming"),
        ({"n": 2}, "acompletion", "multiple_choices"),
        ({"n": 2.0}, "acompletion", "malformed"),
        ({"tools": [{"type": "function", "function": {"name": "f"}}]}, "acompletion", "tool_definitions"),
        ({"functions": [{"name": "f"}]}, "acompletion", "tool_definitions"),
        ({"messages": [{"role": "tool", "tool_call_id": "t", "content": "42"}]}, "acompletion", "tool_message"),
        ({"messages": [{"role": "function", "name": "f", "content": "42"}]}, "acompletion", "tool_message"),
        (
            {"messages": [{"role": "assistant", "content": None, "tool_calls": [{"id": "t"}]}]},
            "acompletion",
            "tool_calls",
        ),
        (
            {"messages": [{"role": "assistant", "content": "x", "function_call": {"name": "f"}}]},
            "acompletion",
            "tool_calls",
        ),
        (
            {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]},
            "acompletion",
            "non_text_input",
        ),
        ({"messages": [{"role": "user", "content": [{"type": "text", "text": 5}]}]}, "acompletion", "non_text_input"),
        ({"messages": [{"role": "user", "content": [{"type": "input_audio"}]}]}, "acompletion", "non_text_input"),
        (
            {"messages": [{"role": "user", "content": [{"type": "image_url", "text": "x"}]}]},
            "acompletion",
            "non_text_input",
        ),  # the part's type decides, not a text field it happens to carry
        (
            {"messages": [{"role": "assistant", "content": "ok", "reasoning_content": "thought"}]},
            "acompletion",
            "reasoning",
        ),
        (
            {"messages": [{"role": "assistant", "content": "ok", "thinking_blocks": [{"type": "thinking"}]}]},
            "acompletion",
            "reasoning",
        ),
        (
            {"messages": [{"role": "assistant", "content": "ok", "reasoning_items": [{"id": "r"}]}]},
            "acompletion",
            "reasoning",
        ),
        ({"messages": [{"role": "assistant", "content": "ok", "reasoning": "thought"}]}, "acompletion", "reasoning"),
        ({"messages": [{"role": "user", "content": None}]}, "acompletion", "content_missing"),
        ({"messages": [{"role": "user", "content": []}]}, "acompletion", "content_missing"),
        ({"messages": [{"role": "user", "content": 5}]}, "acompletion", "malformed"),
        ({"messages": ["hello"]}, "acompletion", "malformed"),
        ({"messages": []}, "acompletion", "malformed"),
        ({"messages": None}, "acompletion", "malformed"),
    ],
)
def test_each_shape_outside_the_profile_has_its_reason(
    change: dict[str, Any], call_type: str | None, reason: str
) -> None:
    assert reason in OUTSIDE_REASONS
    assert request_outside_reason(chat_request(**change), call_type) == reason


def test_a_one_choice_text_response_gives_its_text_from_a_mapping_or_an_object() -> None:
    response = chat_response("the reply")
    assert response_text(response) == ("the reply", None)
    as_object = SimpleNamespace(
        object="chat.completion",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="the reply", tool_calls=None, function_call=None, audio=None, images=None
                )
            )
        ],
    )
    assert response_text(as_object) == ("the reply", None)


def test_empty_reasoning_fields_are_no_reasoning() -> None:
    """LiteLLM drops a reasoning field that is None; one that came empty carries nothing unchecked."""
    empty = {"reasoning_content": "", "thinking_blocks": [], "reasoning_items": [], "reasoning": None}
    assert response_text(chat_response("the reply", **empty)) == ("the reply", None)
    request = chat_request(messages=[{"role": "assistant", "content": "ok", **empty}])
    assert request_outside_reason(request, "acompletion") is None
    with_reasoning = SimpleNamespace(
        object="chat.completion",
        choices=[SimpleNamespace(message=SimpleNamespace(content="safe", reasoning_content="PELICAN"))],
    )
    assert response_text(with_reasoning) == (None, "reasoning")


@pytest.mark.parametrize(
    ("response", "reason"),
    [
        ({**chat_response(), "object": "response"}, "call_type"),
        ({**chat_response(), "object": "chat.completion.chunk"}, "call_type"),
        ({**chat_response(), "choices": []}, "content_missing"),
        ({**chat_response(), "choices": chat_response()["choices"] * 2}, "multiple_choices"),
        ({**chat_response(), "choices": "x"}, "malformed"),
        ({**chat_response(), "choices": [{"index": 0}]}, "content_missing"),
        (chat_response(None, tool_calls=[{"id": "t"}]), "tool_calls"),
        (chat_response("x", function_call={"name": "f"}), "tool_calls"),
        (chat_response("x", audio={"id": "a"}), "non_text_output"),
        (chat_response("x", images=[{"url": "u"}]), "non_text_output"),
        (chat_response("safe", reasoning_content="the code word is PELICAN"), "reasoning"),
        (chat_response("safe", thinking_blocks=[{"type": "thinking", "thinking": "PELICAN"}]), "reasoning"),
        (chat_response("safe", reasoning_items=[{"id": "r", "summary": [{"text": "PELICAN"}]}]), "reasoning"),
        (chat_response("safe", reasoning="PELICAN"), "reasoning"),
        (chat_response(None), "content_missing"),
        (chat_response(["a", "b"]), "content_missing"),
    ],
)
def test_each_response_outside_the_profile_has_its_reason(response: dict[str, Any], reason: str) -> None:
    assert response_text(response) == (None, reason)


# ------------------------------------------------------------------ the request


FACTS = CallFacts(operation_id="op-1", litellm_call_id="call-1", litellm_trace_id="trace-1", model="gpt-4o-mini")
SENDER = Sender(adapter_version="0.1.0", litellm_version="1.102.4", budget_ms=2000)


def test_pre_call_sends_the_messages_as_they_are_and_the_extension() -> None:
    request = chat_request()
    payload = pre_call_payload(request, FACTS, SENDER)
    assert payload == {
        "input_type": "request",
        "structured_messages": request["messages"],
        "texts": None,
        "request_data": {},
        "litellm_call_id": "call-1",
        "litellm_trace_id": "trace-1",
        "model": "gpt-4o-mini",
        "litellm_version": "1.102.4",
        "additional_provider_specific_params": {
            "collieai": {
                "contract_version": 1,
                "adapter": "native",
                "adapter_version": "0.1.0",
                "operation_id": "op-1",
                "source_endpoint": "chat_completions",
                "budget_ms": 2000,
                "stream": False,
                "n": 1,
            }
        },
    }
    assert json.loads(encode(payload)) == payload


def test_post_call_sends_the_one_text_and_no_messages() -> None:
    payload = post_call_payload(chat_request(), "the reply", FACTS, SENDER)
    assert payload["input_type"] == "response"
    assert payload["texts"] == ["the reply"]
    assert "structured_messages" not in payload
    assert (
        payload["additional_provider_specific_params"]
        == pre_call_payload(chat_request(), FACTS, SENDER)["additional_provider_specific_params"]
    )


def test_the_conversation_header_goes_only_when_given_and_nothing_else_of_the_request_does() -> None:
    request = chat_request(user="end-user-7", temperature=0.2, extra_headers={"x-secret": "s"})
    plain = pre_call_payload(request, FACTS, SENDER)
    assert "request_headers" not in plain
    assert not {"user", "temperature", "extra_headers", "metadata"} & set(plain)
    with_id = pre_call_payload(request, CallFacts(operation_id="op-1", conversation_id="conv-9"), SENDER)
    assert with_id["request_headers"] == {"x-collieai-conversation-id": "conv-9"}


def test_a_value_json_cannot_carry_is_refused_by_the_encoder() -> None:
    with pytest.raises((TypeError, ValueError)):
        encode({"x": {1, 2}})
    with pytest.raises(ValueError):
        encode({"x": float("nan")})
