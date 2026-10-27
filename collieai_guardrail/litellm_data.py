"""What the adapter reads from LiteLLM's own data, without importing LiteLLM (LiteLLM 1.102.4).

- The call's type: LiteLLM gives it as a `CallTypes` member or its string value.
- A call's correlation and labels from the request data the proxy hands a hook: `litellm_call_id`, the logging
  object's IDs, the model's name and the conversation ID header the proxy kept (`proxy_server_request.headers`).
- The independent count: LiteLLM's success and failure log events run for every call through its callbacks,
  whether or not a hook did. From their arguments — `call_type`, `messages`, `stream` and `optional_params` — the
  same eligibility rule decides whether the call was one the hooks should have observed. A call the proxy itself
  refused — a bad key, a limit, a budget, or another guardrail's block before the model or after it — is logged
  too, with LiteLLM's marker `litellm_no_upstream_llm_call`; its hooks need not have run, so it is outside,
  `rejected`. A background health check of a model has no messages: outside, `malformed`. A call the router retries
  is logged once per attempt and counted once (`LoggedCalls`): every attempt carries the request's metadata, and in it
  the `operation_id` pre-call issued with its tag. Not by `litellm_call_id`: the proxy takes that from the caller's
  `x-litellm-call-id` header, so two calls can share one.
- The logical call's `operation_id` in the request's metadata, with its tag under this process's secret: issued by
  pre-call, and taken back — by post-call, by the count — only with that tag.

Every function here but `issue_operation_id` reads and never changes what it is given, and none raises on a shape
it does not expect.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import threading
import uuid
from collections import OrderedDict
from collections.abc import Mapping

from collieai_guardrail.counters import Counters
from collieai_guardrail.eligibility import field, request_outside_reason, response_text
from collieai_guardrail.payload import CONVERSATION_HEADER, CallFacts

PRE_CALL = "pre_call"
POST_CALL = "post_call"
#: An identifier longer than this is not sent; the server keeps at most 128 characters of `[A-Za-z0-9._:-]` and
#: counts what it drops.
MAX_ID_CHARS = 256
#: The request fields LiteLLM keeps in `optional_params` that eligibility reads.
_OPTIONAL_FIELDS = ("n", "tools", "functions")
#: LiteLLM's marker of a call its proxy itself refused (`LITELLM_LOGGING_NO_UPSTREAM_LLM_CALL`).
NO_UPSTREAM_CALL = "litellm_no_upstream_llm_call"
REJECTED = "rejected"
#: How many calls' IDs a worker keeps to know a retried attempt from a new call: about a minute at 264 calls a
#: second, longer than the router waits between attempts.
LOGGED_CALLS_KEPT = 16384
#: The key of the logical call's ID in LiteLLM's metadata of the request, and the ID's form: `uuid4().hex`. Beside
#: it, the ID's tag under this process's secret: the caller cannot make one, so an ID with it is one pre-call issued.
OPERATION_ID_KEY = "collieai_operation_id"
OPERATION_TAG_KEY = "collieai_operation_tag"
OPERATION_ID_FORM = re.compile(r"[0-9a-f]{32}")
_OPERATION_SECRET = secrets.token_bytes(32)
#: Where a LiteLLM log event keeps the request's metadata (`litellm_params`): `metadata`, or `litellm_metadata` on
#: the routes that keep the proxy's there.
_LOGGED_METADATA = ("metadata", "litellm_metadata")
#: A refusal mark's reason: one of the closed outside reasons, or empty for an eligible call.
_REASON_FORM = re.compile(r"[a-z_]{0,64}")
#: Enforce: the mark a phase leaves when it refused the call (`mark_refused`), beside the operation's ID.
_REFUSED_KEYS = {PRE_CALL: "collieai_pre_refused", POST_CALL: "collieai_post_refused"}


def call_type_of(value: object) -> str | None:
    """LiteLLM's call type as its string value: a `CallTypes` member (a str enum) or the string itself."""
    value = getattr(value, "value", value)
    return value if isinstance(value, str) and value else None


def call_facts(data: Mapping[str, object], *, operation_id: str) -> CallFacts:
    logging_obj = data.get("litellm_logging_obj")
    return CallFacts(
        operation_id=operation_id,
        litellm_call_id=_id(data.get("litellm_call_id")) or _id(_attr(logging_obj, "litellm_call_id")),
        litellm_trace_id=_id(_attr(logging_obj, "litellm_trace_id")) or _id(data.get("litellm_trace_id")),
        model=_id(data.get("model")),
        conversation_id=conversation_id(data),
    )


def conversation_id(data: Mapping[str, object]) -> str | None:
    """The `x-collieai-conversation-id` header of the caller's request, as the proxy kept it; any case."""
    server_request = data.get("proxy_server_request")
    headers = field(server_request, "headers") if server_request is not None else None
    if not isinstance(headers, Mapping):
        return None
    for name, value in headers.items():
        if isinstance(name, str) and name.lower() == CONVERSATION_HEADER:
            return _id(value)
    return None


def issue_operation_id(metadata: dict[str, object]) -> str:
    """A new logical call's ID, put in the request's metadata with the tag that shows pre-call issued it."""
    operation_id = uuid.uuid4().hex
    metadata[OPERATION_ID_KEY] = operation_id
    metadata[OPERATION_TAG_KEY] = _operation_tag(operation_id)
    return operation_id


def issued_operation_id(metadata: object) -> str | None:
    """The ID in a request's metadata if pre-call in this process issued it, else None. The caller can put anything
    in the request's metadata — a card number in hex has the ID's form — and the ID goes to the server, into LiteLLM's
    log and into the count, so only an ID with the tag of this process's secret is taken."""
    if not isinstance(metadata, Mapping):
        return None
    existing, tag = metadata.get(OPERATION_ID_KEY), metadata.get(OPERATION_TAG_KEY)
    if (
        isinstance(existing, str)
        and isinstance(tag, str)
        and OPERATION_ID_FORM.fullmatch(existing)
        and OPERATION_ID_FORM.fullmatch(tag)  # hex, so the comparison below takes it
        and hmac.compare_digest(tag, _operation_tag(existing))
    ):
        return existing
    return None


def logged_operation_id(kwargs: Mapping[str, object]) -> str | None:
    """The ID pre-call issued for the call a LiteLLM log event is for, from the request's metadata logged with it."""
    params = kwargs.get("litellm_params")
    if not isinstance(params, Mapping):
        return None
    for key in _LOGGED_METADATA:
        operation_id = issued_operation_id(params.get(key))
        if operation_id is not None:
            return operation_id
    return None


def mark_refused(metadata: dict[str, object], phase: str, operation_id: str, reason: str | None) -> None:
    """Enforce: this phase refused the call. LiteLLM logs it as a failure — before the model a call its proxy refused
    (`rejected`), after it one that never succeeded — so the count takes the phase as seen, by `reason` (None: the
    call was eligible), as it would have had the call gone on: else `logged_unseen` would drift by every refusal.
    Tagged as the operation's ID is, so only a mark this process made is taken."""
    value = reason or ""
    metadata[_REFUSED_KEYS[phase]] = f"{value}:{_operation_tag(f'refused:{phase}:{operation_id}:{value}')}"


def refused_by_us(kwargs: Mapping[str, object], phase: str, operation_id: str | None) -> tuple[str | None] | None:
    """The mark `mark_refused` left on the call a LiteLLM log event is for: `(reason,)`, or None without one."""
    if operation_id is None:
        return None
    params = kwargs.get("litellm_params")
    if not isinstance(params, Mapping):
        return None
    for key in _LOGGED_METADATA:
        metadata = params.get(key)
        mark = metadata.get(_REFUSED_KEYS[phase]) if isinstance(metadata, Mapping) else None
        if not isinstance(mark, str) or ":" not in mark:
            continue
        value, tag = mark.rsplit(":", 1)
        if not _REASON_FORM.fullmatch(value):  # not a mark this process wrote: never tagged, never counted
            continue
        expected = _operation_tag(f"refused:{phase}:{operation_id}:{value}")
        if OPERATION_ID_FORM.fullmatch(tag) and hmac.compare_digest(tag, expected):
            return (value or None,)
    return None


def _operation_tag(operation_id: str) -> str:
    return hmac.new(_OPERATION_SECRET, operation_id.encode("ascii"), hashlib.sha256).hexdigest()[:32]


def logged_request(kwargs: Mapping[str, object]) -> dict[str, object]:
    """The request's fields that eligibility reads, from a LiteLLM log event's arguments."""
    request: dict[str, object] = {"messages": kwargs.get("messages"), "stream": kwargs.get("stream")}
    optional = kwargs.get("optional_params")
    if isinstance(optional, Mapping):
        for name in _OPTIONAL_FIELDS:
            if name in optional:
                request[name] = optional[name]
    return request


class LoggedCalls:
    """The calls LiteLLM logged lately, by the `operation_id` pre-call issued. Its router logs every attempt of a
    call it retries or falls back for — a failure, then the next attempt's outcome — with the request's one metadata,
    so a call's pre-call phase is counted at its first event and its post-call phase at its success, once each.
    Bounded, oldest first out; the IDs stay in this process's memory and go nowhere."""

    def __init__(self, capacity: int = LOGGED_CALLS_KEPT):
        self._capacity = capacity
        self._calls: OrderedDict[str, bool] = OrderedDict()  # the call's ID → its post-call phase was counted
        self._lock = threading.Lock()

    def see(self, call_id: str, *, succeeded: bool) -> tuple[bool, bool]:
        """Whether this event is the call's first, and whether it is the call's first success."""
        with self._lock:
            known = call_id in self._calls
            counted_post = self._calls.get(call_id, False)
            self._calls[call_id] = counted_post or succeeded
            self._calls.move_to_end(call_id)
            while len(self._calls) > self._capacity:
                self._calls.popitem(last=False)
        return not known, succeeded and not counted_post


def count_logged(
    counters: Counters,
    kwargs: Mapping[str, object],
    response: object,
    *,
    succeeded: bool,
    calls: LoggedCalls | None = None,
) -> None:
    """Count one event LiteLLM logged. Every logged call had its pre-call phase; only a successful one had a
    post-call phase. A call outside the supported profile is counted by its reason. With `calls`, a further event of a
    call already counted — a retried attempt, known by pre-call's `operation_id` — counts as `logged_repeats` and
    nothing else. An event without that ID is a call of its own: pre-call did not run for it, and a hook that did not
    run must show."""
    call_id = logged_operation_id(kwargs)
    pre_refused = None if succeeded else refused_by_us(kwargs, PRE_CALL, call_id)
    post_refused = None if succeeded else refused_by_us(kwargs, POST_CALL, call_id)
    seen_post = succeeded or post_refused is not None
    first, first_success = calls.see(call_id, succeeded=seen_post) if calls and call_id else (True, seen_post)
    if not first:
        counters.add(PRE_CALL, "logged_repeats")
    reason: str | None
    try:
        if pre_refused is not None:  # our own refusal before the model, not another's
            reason = pre_refused[0]
        elif kwargs.get(NO_UPSTREAM_CALL):
            reason = REJECTED
        else:
            reason = request_outside_reason(logged_request(kwargs), call_type_of(kwargs.get("call_type")))
    except Exception:
        reason = "malformed"
    if first:
        _count(counters, PRE_CALL, reason)
    if not first_success:
        return
    if reason is None:
        if post_refused is not None:  # refused after the model: the response never reached the log
            reason = post_refused[0]
        else:
            try:
                _, reason = response_text(response)
            except Exception:
                reason = "malformed"
    _count(counters, POST_CALL, reason)


def _count(counters: Counters, phase: str, reason: str | None) -> None:
    counters.add(phase, "logged_calls")
    if reason is None:
        counters.add(phase, "logged_eligible")
    else:
        counters.add_reason(phase, "logged_outside", reason)


def _attr(obj: object, name: str) -> object:
    try:
        return getattr(obj, name, None)
    except Exception:
        return None


def _id(value: object) -> str | None:
    return value if isinstance(value, str) and value and len(value) <= MAX_ID_CHARS else None
