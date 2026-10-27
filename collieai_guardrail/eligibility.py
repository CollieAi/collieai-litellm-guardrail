"""Which calls are in the supported profile.

The supported profile is foreground, non-streaming Chat Completions text with one choice. Anything else is passed
through unchanged with no request sent, and counted as outside under one closed reason code — it is not part of the
coverage. The server's 422 `shape` stays a second line, not the main path.

Both functions read the call; neither changes it. They accept LiteLLM's objects as well as plain mappings: a field
is read by key from a mapping and by attribute from anything else.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

#: LiteLLM's call types of a Chat Completion (`CallTypesLiteral`); the proxy's chat route uses `acompletion`.
CHAT_CALL_TYPES = frozenset({"acompletion", "completion"})

#: In `block` the only calls that pass unchecked: they generate no text a policy could block. Every other call outside
#: the profile — a stream, tools, Responses, Messages, a pass-through route, a type LiteLLM adds later — is refused
#: before the model: else switching the endpoint, or streaming, would get past the policy.
UNCHECKED_CALL_TYPES = frozenset({"aembedding", "embedding", "arerank", "rerank", "amoderation", "moderation"})

#: The closed reasons a call is outside the supported profile.
OUTSIDE_REASONS = (
    "call_type",  # not a Chat Completion: Responses, embeddings, images, a pass-through route...
    "streaming",
    "multiple_choices",  # n > 1, or a response with more than one choice
    "tool_definitions",  # tools or functions offered to the model
    "tool_calls",  # a message or the response carries tool or function calls
    "tool_message",  # a tool's or a function's message in the history
    "non_text_input",  # a content part that is not text
    "non_text_output",  # the response carries audio or images
    "content_missing",  # a message or the response without content: null, no parts, no choice — "" is content
    "reasoning",  # a message or the response carries the model's reasoning, which is not checked
    "malformed",  # not the shape of a Chat Completion at all
)

#: A message's fields with the model's reasoning in the pinned LiteLLM (`Message`, `ChatCompletionAssistantMessage`);
#: `reasoning` is a provider's name for `reasoning_content`. Only `content` is sent, so a call that carries any of
#: them is not wholly checked — outside rather than a check that looks complete.
REASONING_FIELDS = ("reasoning_content", "thinking_blocks", "reasoning_items", "reasoning")


def field(obj: object, name: str) -> object:
    """A field of a mapping by key, of anything else by attribute; None when absent."""
    if isinstance(obj, Mapping):
        return obj.get(name)
    return getattr(obj, name, None)


def request_outside_reason(request: Mapping[str, object], call_type: str | None) -> str | None:
    """Why a request is outside the supported profile, or None when it is in it. A call whose type is not known
    (None) is outside: a chat-shaped request does not make a call a Chat Completion."""
    if call_type not in CHAT_CALL_TYPES:
        return "call_type"
    if request.get("stream"):
        return "streaming"
    n = request.get("n")
    if n is not None and type(n) is not int:
        return "malformed"
    if n is not None and n != 1:
        return "multiple_choices"
    if request.get("tools") or request.get("functions"):
        return "tool_definitions"
    messages = request.get("messages")
    if not isinstance(messages, list) or not messages:
        return "malformed"
    for message in messages:
        reason = _message_outside_reason(message)
        if reason is not None:
            return reason
    return None


def output_outside_reason(request: Mapping[str, object]) -> str | None:
    """Why the reply the request asks for could not be checked after the model — audio or another output than text —
    or None. Only `block` reads it: a reply post-call cannot check must be refused before the model is called."""
    modalities = request.get("modalities")
    if request.get("audio") or (isinstance(modalities, list) and any(m != "text" for m in modalities)):
        return "non_text_output"
    if modalities is not None and not isinstance(modalities, list):
        return "malformed"
    return None


def _message_outside_reason(message: object) -> str | None:
    if not isinstance(message, Mapping):
        return "malformed"
    if message.get("role") in ("tool", "function"):
        return "tool_message"
    if message.get("tool_calls") or message.get("function_call"):
        return "tool_calls"
    if _carries_reasoning(message):
        return "reasoning"
    content = message.get("content")
    if isinstance(content, str):
        return None
    if content is None:
        return "content_missing"
    if not isinstance(content, list):
        return "malformed"
    if not content:
        return "content_missing"
    for part in content:
        if not isinstance(part, Mapping) or part.get("type") != "text" or not isinstance(part.get("text"), str):
            return "non_text_input"
    return None


def response_text(response: object) -> tuple[str | None, str | None]:
    """The text of a one-choice Chat Completion response, or why the response is outside the supported profile:
    `(text, None)` or `(None, reason)`."""
    if field(response, "object") != "chat.completion":
        return None, "call_type"
    choices = field(response, "choices")
    if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)):
        return None, "malformed"
    if len(choices) > 1:
        return None, "multiple_choices"
    if not choices:
        return None, "content_missing"
    message = field(choices[0], "message")
    if message is None:
        return None, "content_missing"
    if field(message, "tool_calls") or field(message, "function_call"):
        return None, "tool_calls"
    if field(message, "audio") or field(message, "images"):
        return None, "non_text_output"
    if _carries_reasoning(message):
        return None, "reasoning"
    content = field(message, "content")
    if not isinstance(content, str):
        return None, "content_missing"
    return content, None  # an empty text is a text: the server judges it as an item


def _carries_reasoning(message: object) -> bool:
    return any(field(message, name) for name in REASONING_FIELDS)
