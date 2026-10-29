"""The LiteLLM glue: CollieAi's Monitor observation as a LiteLLM `CustomGuardrail` (LiteLLM 1.102.4).

The operator loads it from the proxy's config (`guardrail: collieai_guardrail.guardrail.CollieAIGuardrail`, `mode:
[pre_call, post_call]`); LiteLLM's registry builds it with the guardrail's `litellm_params` as keyword arguments and
adds it to its callbacks. It defines no `apply_guardrail`, so the proxy calls its own hooks:

- `async_pre_call_hook` and `async_post_call_success_hook` observe the phase through the core (`observer`) and
  return None — LiteLLM keeps the request and the response as they are. What happened goes into LiteLLM's standard
  guardrail log (`StandardLoggingGuardrailInformation`) with its status: `success`, `guardrail_flagged`,
  `guardrail_failed_to_respond` or `not_run`; a completed check also with `guardrail_action` and
  `violation_categories`, through LiteLLM's `tracing_detail` — the fields its logs keep where they redact the
  guardrail's response.
- Pre-call gives the logical call its `operation_id` in LiteLLM's metadata of the request, with a tag under the
  process's secret; post-call reads it back only with that tag, so both phases carry the same one and an ID the
  caller wrote there is never taken for it. Post-call learns the call's type from LiteLLM's logging object, or else
  from the request's route as LiteLLM's own guardrails do.
- `async_log_success_event` and `async_log_failure_event` count every call LiteLLM logs, apart from the hooks
  (`litellm_data.count_logged`): a hook that does not run still shows. A call the router retries is counted once, by
  pre-call's `operation_id` in its metadata.
- The heartbeat (`heartbeat`) logs the counts and runs the synthetic check on the worker's event loop. When LiteLLM
  drops the instance from its callbacks — the guardrail deleted, or rebuilt with changed settings — the instance
  retires: its heartbeat stops and, a deadline later, its client closes.

With `enforcement_mode: block` (enforce with blocking only) a hook refuses the call the outcome says to refuse
(`Outcome.refusal`): it records the phase first, then raises
LiteLLM's `GuardrailRaisedException` with the status — 400 for a block or a call outside what the adapter checks
(nothing reaches the model before the call; the model's text never reaches the client after it), 503 when the
check could not be made. A fault of the glue itself refuses the call too. Block needs both phases, `default_on` and
the readiness guard among the proxy's callbacks (`RequireGuardrail`), which refuses every call while no CollieAi
guardrail is loaded.

A setting the adapter cannot run with stops the proxy's start (`ConfigError`), `mode` included whatever
LITELLM_STRICT_GUARDRAIL_MODES says. Past that, in Monitor nothing here raises into LiteLLM but a cancellation of the
call; a fault of the glue itself is logged and the call goes on unchanged. Unless the operator configured the
`collieai_guardrail` logger, the adapter makes its INFO lines show (`_make_visible`).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Mapping
from importlib import metadata
from typing import Any

import httpx
import litellm
from litellm.exceptions import GuardrailRaisedException
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.api_route_to_call_types import get_call_types_for_route
from litellm.litellm_core_utils.core_helpers import get_or_create_metadata_bucket
from litellm.types.guardrails import GuardrailEventHooks, Mode
from litellm.types.utils import GuardrailTracingDetail

from collieai_guardrail.config import (
    DEFAULT_DEADLINE_MS,
    DEFAULT_MAX_BODY_BYTES,
    DEFAULT_NETWORK_MARGIN_MS,
    DEFAULT_SUMMARY_INTERVAL_S,
    DEFAULT_SYNTHETIC_INTERVAL_S,
    AdapterConfig,
    ConfigError,
    guardrail_name,
    named,
)
from collieai_guardrail.counters import Counters
from collieai_guardrail.eligibility import request_outside_reason
from collieai_guardrail.heartbeat import Heartbeat
from collieai_guardrail.litellm_data import (
    LoggedCalls,
    call_facts,
    call_type_of,
    count_logged,
    count_refused,
    issue_operation_id,
    issued_operation_id,
    mark_refused,
)
from collieai_guardrail.observer import UNAVAILABLE_CODE, Observer, Outcome

logger = logging.getLogger("collieai_guardrail")

PRE_CALL = GuardrailEventHooks.pre_call.value
POST_CALL = GuardrailEventHooks.post_call.value
PROVIDER = "collieai"
HOOKS = (GuardrailEventHooks.pre_call, GuardrailEventHooks.post_call)
OUR_HOOKS = frozenset(hook.value for hook in HOOKS)
#: What a refused call's client reads, by the refusal's code: fixed text, the code in it — nothing of the policy's.
REFUSAL_MESSAGES = {
    "content_blocked": "Request blocked by the CollieAi policy",
    "response_blocked": "Response blocked by the CollieAi policy",
    "unsupported_request": "Request outside what the CollieAi guardrail checks",
    "unsupported_response": "Response outside what the CollieAi guardrail checks",
    UNAVAILABLE_CODE: "CollieAi guardrail unavailable",
}
NOT_LOADED_CODE = "guardrail_not_loaded"
#: The readiness guard as `litellm_settings.callbacks` names it.
GUARD_PATH = "collieai_guardrail.guardrail.require_guardrail"
#: The name LiteLLM's record of a call the guard refused gives it.
GUARD_NAME = "collieai-readiness"
#: The least time between two of the guard's ERROR lines in one worker.
GUARD_LOG_INTERVAL_S = 60.0


class CollieAIGuardrail(CustomGuardrail):
    _collieai_kind = "guardrail"  # what `_is_collieai` knows it by

    def __init__(
        self,
        *,
        api_base: str | None = None,
        api_key: str | None = None,
        deadline_ms: int = DEFAULT_DEADLINE_MS,
        network_margin_ms: int = DEFAULT_NETWORK_MARGIN_MS,
        max_body_bytes: int = DEFAULT_MAX_BODY_BYTES,
        forward_conversation_id: bool = False,
        summary_interval_s: int = DEFAULT_SUMMARY_INTERVAL_S,
        synthetic_interval_s: int = DEFAULT_SYNTHETIC_INTERVAL_S,
        enforcement_mode: str = "monitor",
        client: httpx.AsyncClient | None = None,
        **kwargs: Any,  # LiteLLM's own parameters, of many types, for CustomGuardrail
    ) -> None:
        """The keywords are the guardrail's `litellm_params`. LiteLLM also passes its own (`guardrail_name`,
        `event_hook`, `default_on` and the defaults of every guardrail's parameters); they go to `CustomGuardrail`.
        `client` is for embedding and tests; the adapter builds its own otherwise."""
        try:
            self.config = AdapterConfig(
                api_base=api_base,  # type: ignore[arg-type]  # AdapterConfig refuses a missing one by name
                api_key=api_key,  # type: ignore[arg-type]
                deadline_ms=deadline_ms,
                network_margin_ms=network_margin_ms,
                max_body_bytes=max_body_bytes,
                forward_conversation_id=forward_conversation_id,
                summary_interval_s=summary_interval_s,
                synthetic_interval_s=synthetic_interval_s,
                enforcement_mode=enforcement_mode,
            )
            _check_mode(kwargs.get("event_hook"))
            if self.config.enforce:
                _check_enforce_wiring(kwargs.get("event_hook"), kwargs.get("default_on"))
                _check_guard()
            super().__init__(supported_event_hooks=list(HOOKS), **kwargs)
        except (ValueError, TypeError) as error:  # LiteLLM's loader would swallow it and start without the guardrail
            raise ConfigError(f"the guardrail's settings were refused ({type(error).__name__})") from error
        _make_visible(logger)
        name = guardrail_name(self.guardrail_name)
        self._named = named(name)
        self.counters = Counters()
        self._logged_calls = LoggedCalls()
        self.observer = Observer(
            self.config, litellm_version=_litellm_version(), client=client, counters=self.counters, name=name
        )
        self._owns_client = client is None
        self._registered = False
        self.enforce = self.config.enforce
        self.heartbeat = Heartbeat(
            self.observer,
            self.counters,
            summary_interval_s=self.config.summary_interval_s,
            synthetic_interval_s=self.config.synthetic_interval_s,
            retired=self._retired,
            on_retired=self._release,
            name=name,
        )
        self._warn_about_wiring()
        self.heartbeat.ensure_running()

    def apply_stream_scope(self, stream_scope: object) -> None:
        """LiteLLM's `stream_scope` (1.106; 1.102 has none) runs a phase only for streaming calls, or only for the
        others: in block the rest would go through unchecked, so block starts only without one or with `both`. LiteLLM
        applies it to the guardrail it has built, and calls this from its constructor too."""
        if self.config.enforce and stream_scope not in (None, "both"):
            raise ConfigError("enforcement_mode block needs stream_scope unset or both")
        apply = getattr(super(), "apply_stream_scope", None)
        if apply is not None:
            apply(stream_scope)

    # ------------------------------------------------------------------ the hooks

    async def async_pre_call_hook(
        self, user_api_key_dict: object, cache: object, data: dict[str, object], call_type: object
    ) -> None:
        self.heartbeat.ensure_running()
        outcome: Outcome | None = None
        try:
            started = time.time()
            operation_id = _issue_operation_id(data)  # a new logical call, whatever the request's metadata says
            facts = call_facts(data, operation_id=operation_id)
            outcome = await self.observer.observe_request(data, call_type=call_type_of(call_type), facts=facts)
            self._record(data, outcome, GuardrailEventHooks.pre_call, started, operation_id)
            if outcome.refusal is not None:
                self._mark_refused(PRE_CALL, data, operation_id, outcome)
        except Exception as error:  # a cancellation goes on; nothing else reaches LiteLLM
            self._glue_failed(PRE_CALL, error, data, outcome)
            return None
        self._refuse_if_asked(outcome)
        return None

    async def async_post_call_success_hook(
        self, data: dict[str, object], user_api_key_dict: object, response: object
    ) -> None:
        self.heartbeat.ensure_running()
        outcome: Outcome | None = None
        call_type: str | None = None
        try:
            started = time.time()
            operation_id = _operation_id(data)
            facts = call_facts(data, operation_id=operation_id)
            call_type = _post_call_type(data, user_api_key_dict)
            outcome = await self.observer.observe_response(data, response, call_type=call_type, facts=facts)
            self._record(data, outcome, GuardrailEventHooks.post_call, started, operation_id)
            if outcome.refusal is not None:
                self._mark_refused(POST_CALL, data, operation_id, outcome, call_type=call_type)
        except Exception as error:
            self._glue_failed(POST_CALL, error, data, outcome, call_type=call_type)
            return None
        self._refuse_if_asked(outcome)
        return None

    def _glue_failed(
        self,
        phase: str,
        error: Exception,
        data: dict[str, object],
        outcome: Outcome | None,
        *,
        call_type: str | None = None,
    ) -> None:
        """A fault of the glue itself: in Monitor the call goes on; in block it is refused, as a check not made — with
        its mark when the call has its ID, so the count still sees it under the reason the outcome had, if any."""
        logger.warning(
            "collieai guardrail %s: the glue failed (%s); %s%s",
            phase,
            type(error).__name__,
            "the call is refused (enforcement_mode: block)" if self.enforce else "the call goes on",
            self._named,
        )
        if self.enforce:
            try:
                operation_id = issued_operation_id(_metadata(data))
                if operation_id is not None:
                    self._mark_refused(phase, data, operation_id, outcome, call_type=call_type)
            except Exception:  # the refusal stands; only the count may miss it
                pass
            raise self._refusal(503, UNAVAILABLE_CODE, blocked_content=False) from None

    def _mark_refused(
        self,
        phase: str,
        data: dict[str, object],
        operation_id: str,
        outcome: Outcome | None,
        *,
        call_type: str | None = None,
    ) -> None:
        """The refusal's mark, and its count now: LiteLLM's event about it, if one comes, is skipped."""
        reason = _counted_reason(outcome)
        mark_refused(_metadata(data), phase, operation_id, reason)
        request_reason = request_outside_reason(data, call_type) if phase == POST_CALL else None
        count_refused(
            self.counters,
            phase,
            reason,
            operation_id=operation_id,
            request_reason=request_reason,
            calls=self._logged_calls,
        )

    def _refuse_if_asked(self, outcome: Outcome) -> None:
        """Block only: refuse the call as the outcome says, after it was recorded. Never in Monitor."""
        if not self.enforce or outcome.refusal is None:
            return
        status, code = outcome.refusal
        raise self._refusal(status, code, blocked_content=outcome.check is not None and outcome.check.blocked)

    def _refusal(self, status: int, code: str, *, blocked_content: bool) -> GuardrailRaisedException:
        """LiteLLM's own refusal of a guardrail: the client gets `status` and the fixed text with the code."""
        return GuardrailRaisedException(
            guardrail_name=self.guardrail_name,
            message=f"{REFUSAL_MESSAGES[code]} ({code})",
            should_wrap_with_default_message=False,
            status_code=status,
            blocked_content=blocked_content,
        )

    # ------------------------------------------------------------------ the count apart from the hooks

    async def async_log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        self._count_logged(kwargs, response_obj, succeeded=True)

    async def async_log_failure_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        self._count_logged(kwargs, response_obj, succeeded=False)

    def _count_logged(self, kwargs: Mapping[str, object], response: object, *, succeeded: bool) -> None:
        self.heartbeat.ensure_running()
        try:
            count_logged(self.counters, kwargs, response, succeeded=succeeded, calls=self._logged_calls)
        except Exception as error:
            logger.warning(
                "collieai guardrail: counting a logged call failed (%s)%s", type(error).__name__, self._named
            )

    # ------------------------------------------------------------------ helpers

    def _record(
        self, data: dict[str, object], outcome: Outcome, event: GuardrailEventHooks, started: float, operation_id: str
    ) -> None:
        self.add_standard_logging_guardrail_information_to_request_data(
            guardrail_json_response={**outcome.diagnostics(), "operation_id": operation_id},
            request_data=data,
            guardrail_status=outcome.guardrail_status,
            start_time=started,
            end_time=started + outcome.duration_s,
            duration=outcome.duration_s,
            guardrail_provider=PROVIDER,
            event_type=event,
            tracing_detail=_tracing_detail(outcome),
        )

    def _retired(self) -> bool:
        """LiteLLM no longer has this instance among its callbacks: the guardrail was deleted, or rebuilt as a new
        instance with changed settings.

        An instance that built its own client is LiteLLM's: its registry adds it to the callbacks in the same step that
        builds it (`initialize_custom_guardrail`), before the heartbeat's first turn can run, so its absence means a
        removal — also one that came before that turn. An instance given its client is embedded: it retires only once
        LiteLLM had it and dropped it, and one LiteLLM never registered never retires."""
        registered = any(callback is self for callback in litellm.callbacks)
        self._registered = self._registered or registered
        return not registered and (self._registered or self._owns_client)

    async def _release(self) -> None:
        """After retiring: a hook LiteLLM dispatched before the removal ends within the deadline; then the client
        the adapter built is closed."""
        await asyncio.sleep(self.config.deadline_ms / 1000)
        if self._owns_client:
            await self.observer.aclose()

    def _warn_about_wiring(self) -> None:
        if self.enforce:  # refused at the start instead (`_check_enforce_wiring`)
            return
        hooks = _hooks(self.event_hook)
        for hook in HOOKS:
            if hook.value not in hooks:
                logger.warning(
                    "collieai guardrail: mode has no %s; that phase is not observed"
                    " (set mode: [pre_call, post_call])%s",
                    hook.value,
                    self._named,
                )
        if not self.default_on:
            logger.warning(
                "collieai guardrail: default_on is false; only requests that name the guardrail are observed%s",
                self._named,
            )


class RequireGuardrail(CustomLogger):
    """The readiness guard for `enforcement_mode: block`, one line in the proxy's config:

        litellm_settings:
          callbacks: ["collieai_guardrail.guardrail.require_guardrail"]

    LiteLLM starts without a guardrail whose entry it refuses — a missing `mode`, a value of the wrong type:
    `init_guardrails_v2` logs "Skipping guardrail" and goes on, and calls go through unchecked. Its checks of the entry
    fail before it builds the guardrail, so the guardrail cannot stop that start. The guard can: LiteLLM loads the
    callbacks before the guardrails and does not swallow an error loading them. It refuses every call (503
    `guardrail_not_loaded`) while the worker has no CollieAi guardrail in block among its callbacks, and the guardrail
    in block does not start without it (`_check_guard`), so a mistake in the entry refuses the calls instead.

    Only a guardrail in block counts. A misspelled `enforcement_mode` key reaches the adapter as an unknown setting and
    leaves it in Monitor; the guard refuses then too, instead of the proxy observing quietly. Going back to Monitor
    takes the guard's line out as well. The guard reads no request and sends nothing."""

    _collieai_kind = "readiness"  # what `_is_collieai` knows it by
    # LiteLLM runs a plain callback's pre-call hook in its guardrail-only scans (a batch file's records; the MCP tool
    # catalog in 1.106) only when the callback says it judges the request.
    enforces_request_content = True

    def __init__(self) -> None:
        super().__init__()
        self._named = named(GUARD_NAME)
        self._reported_at: float | None = None

    async def async_pre_call_hook(
        self, user_api_key_dict: object, cache: object, data: dict[str, object], call_type: object
    ) -> None:
        if any(
            _is_collieai(callback, "guardrail") and getattr(callback, "enforce", False) is True
            for callback in litellm.callbacks
        ):
            return None
        self._report()
        raise GuardrailRaisedException(
            guardrail_name=GUARD_NAME,
            message=f"CollieAi guardrail not loaded ({NOT_LOADED_CODE})",
            should_wrap_with_default_message=False,
            status_code=503,
        )

    def _report(self) -> None:
        now = time.monotonic()
        if self._reported_at is not None and now - self._reported_at < GUARD_LOG_INTERVAL_S:
            return
        self._reported_at = now
        _make_visible(logger)  # no guardrail may have been built in this worker to do it
        logger.error(
            "collieai guardrail readiness: no CollieAi guardrail in enforcement_mode block is loaded in this worker;"
            " every call is refused (503 %s). LiteLLM logs a guardrail it skips at the start ('Skipping guardrail');"
            " to go back to Monitor, take %s out of litellm_settings.callbacks as well%s",
            NOT_LOADED_CODE,
            GUARD_PATH,
            self._named,
        )


#: The instance `litellm_settings.callbacks` names (`GUARD_PATH`).
require_guardrail = RequireGuardrail()


def _counted_reason(outcome: Outcome | None) -> str | None:
    """The reason the count takes a refused phase under: an outside call's, None for an eligible one."""
    return outcome.reason if outcome is not None and outcome.result == "outside" else None


def _tracing_detail(outcome: Outcome) -> GuardrailTracingDetail | None:
    """A completed check's action and what it found, for the fields of LiteLLM's record its logs keep as they are;
    none where no check was made."""
    if outcome.guardrail_action is None:
        return None
    return GuardrailTracingDetail(
        guardrail_action=outcome.guardrail_action, violation_categories=outcome.violation_categories
    )


def _metadata(data: dict[str, object]) -> dict[str, object]:
    _, bucket = get_or_create_metadata_bucket(data)
    return bucket


def _issue_operation_id(data: dict[str, object]) -> str:
    return issue_operation_id(_metadata(data))


def _operation_id(data: dict[str, object]) -> str:
    """Pre-call's ID for this call — only one with its tag (`litellm_data.issued_operation_id`); a new one when
    pre-call did not run for it."""
    bucket = _metadata(data)
    return issued_operation_id(bucket) or issue_operation_id(bucket)


def _post_call_type(data: Mapping[str, object], user_api_key_dict: object) -> str | None:
    """LiteLLM does not give post-call the call's type: its logging object has it; else the request's route."""
    call_type = call_type_of(getattr(data.get("litellm_logging_obj"), "call_type", None))
    if call_type is not None:
        return call_type
    route = getattr(user_api_key_dict, "request_route", None)
    if not isinstance(route, str) or not route:
        return None
    call_types = get_call_types_for_route(route)
    return call_type_of(call_types[0]) if call_types else None


def _named_hooks(event_hook: object) -> list[object]:
    """What a guardrail's `mode` names: one hook, a list, or a tag-based `Mode`'s tags and default. LiteLLM runs a
    guardrail without a mode on every hook (`CustomGuardrail._event_hook_is_event_type`): both of ours."""
    if event_hook is None:
        return [hook.value for hook in HOOKS]
    if isinstance(event_hook, Mode):
        named: list[object] = []
        for value in (*event_hook.tags.values(), event_hook.default or []):
            named.extend(value if isinstance(value, list) else [value])
        return named
    return list(event_hook) if isinstance(event_hook, (list, tuple, set)) else [event_hook]


def _check_mode(event_hook: object) -> None:
    """The adapter's own check of `mode`. LiteLLM's (`CustomGuardrail._validate_event_hook`) only warns under
    LITELLM_STRICT_GUARDRAIL_MODES=false, and the guardrail would load with no hook of its own and observe nothing."""
    named = [call_type_of(hook) for hook in _named_hooks(event_hook)]
    if not named or any(hook not in OUR_HOOKS for hook in named):
        raise ConfigError("mode must be pre_call, post_call or both")


def _check_enforce_wiring(event_hook: object, default_on: object) -> None:
    """Block refuses what it could not check, so a phase it does not see, or a request that does not name the
    guardrail, would go through unchecked: both phases and `default_on: true` or the proxy does not start. A mode by
    tags runs a phase only for the tags it names, and a request's tags are the caller's to send: not in block."""
    if isinstance(event_hook, Mode):
        raise ConfigError("enforcement_mode block needs mode: [pre_call, post_call], not a mode by tags")
    if _hooks(event_hook) != set(OUR_HOOKS):
        raise ConfigError("enforcement_mode block needs mode: [pre_call, post_call]")
    if default_on is not True:
        raise ConfigError("enforcement_mode block needs default_on: true")


def _is_collieai(callback: object, kind: str) -> bool:
    """The guardrail or the guard by its class's mark, not `isinstance`: LiteLLM loads a module that sits next to its
    config by its file path, so the two may come from two copies of this module."""
    return getattr(type(callback), "_collieai_kind", None) == kind


def _check_guard() -> None:
    """Block also needs the readiness guard among the proxy's callbacks (`RequireGuardrail`): without it, an entry
    LiteLLM refuses before building the guardrail would leave every call unchecked. LiteLLM loads the callbacks
    first."""
    if not any(_is_collieai(callback, "readiness") for callback in litellm.callbacks):
        raise ConfigError(
            f'enforcement_mode block needs the readiness guard: add "{GUARD_PATH}" to litellm_settings.callbacks'
        )


def _hooks(event_hook: object) -> set[str]:
    return {value for value in (call_type_of(hook) for hook in _named_hooks(event_hook)) if value is not None}


def _make_visible(log: logging.Logger) -> None:
    """The proxy gives handlers to LiteLLM's own loggers and leaves the root at WARNING without one, so the adapter's
    INFO — the summary, a passing synthetic check — would show nowhere. Unless the operator set a level for
    `collieai_guardrail`, it logs from INFO. Unless a handler up its chain takes INFO (`json_logs: true` in the proxy's
    `litellm_settings` gives the root one that does), it gets its own, on stderr, for the levels below the lowest one
    the chain's handlers take: a record goes out once. Its own takes every level itself (NOTSET), so the next instance
    finds INFO taken."""
    if log.level == logging.NOTSET:
        log.setLevel(logging.INFO)
    taken_from = _lowest_level_taken(log)
    if taken_from is not None and taken_from <= logging.INFO:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    if taken_from is not None:
        handler.addFilter(lambda record: record.levelno < taken_from)
    log.addHandler(handler)


def _lowest_level_taken(log: logging.Logger) -> int | None:
    """The lowest level a handler takes of the logger's records, up its chain as `Logger.callHandlers` walks it;
    None when there is no handler at all."""
    levels: list[int] = []
    current: logging.Logger | None = log
    while current is not None:
        levels.extend(handler.level for handler in current.handlers)
        current = current.parent if current.propagate else None
    return min(levels) if levels else None


def _litellm_version() -> str | None:
    try:
        return metadata.version("litellm")
    except metadata.PackageNotFoundError:
        return None
