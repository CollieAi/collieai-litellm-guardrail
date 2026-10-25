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

A setting the adapter cannot run with stops the proxy's start (`ConfigError`), `mode` included whatever
LITELLM_STRICT_GUARDRAIL_MODES says. Past that, nothing here raises into LiteLLM but a cancellation of the call; a
fault of the glue itself is logged and the call goes on unchanged. Unless the operator configured the
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
from litellm.integrations.custom_guardrail import CustomGuardrail
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
from collieai_guardrail.heartbeat import Heartbeat
from collieai_guardrail.litellm_data import (
    LoggedCalls,
    call_facts,
    call_type_of,
    count_logged,
    issue_operation_id,
    issued_operation_id,
)
from collieai_guardrail.observer import Observer, Outcome

logger = logging.getLogger("collieai_guardrail")

PROVIDER = "collieai"
HOOKS = (GuardrailEventHooks.pre_call, GuardrailEventHooks.post_call)
OUR_HOOKS = frozenset(hook.value for hook in HOOKS)


class CollieAIGuardrail(CustomGuardrail):
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
            )
            _check_mode(kwargs.get("event_hook"))
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

    # ------------------------------------------------------------------ the hooks

    async def async_pre_call_hook(
        self, user_api_key_dict: object, cache: object, data: dict[str, object], call_type: object
    ) -> None:
        self.heartbeat.ensure_running()
        try:
            started = time.time()
            operation_id = _issue_operation_id(data)  # a new logical call, whatever the request's metadata says
            facts = call_facts(data, operation_id=operation_id)
            outcome = await self.observer.observe_request(data, call_type=call_type_of(call_type), facts=facts)
            self._record(data, outcome, GuardrailEventHooks.pre_call, started, operation_id)
        except Exception as error:  # a cancellation goes on; nothing else reaches LiteLLM
            logger.warning(
                "collieai guardrail pre_call: the glue failed (%s); the call goes on%s",
                type(error).__name__,
                self._named,
            )
        return None

    async def async_post_call_success_hook(
        self, data: dict[str, object], user_api_key_dict: object, response: object
    ) -> None:
        self.heartbeat.ensure_running()
        try:
            started = time.time()
            operation_id = _operation_id(data)
            facts = call_facts(data, operation_id=operation_id)
            call_type = _post_call_type(data, user_api_key_dict)
            outcome = await self.observer.observe_response(data, response, call_type=call_type, facts=facts)
            self._record(data, outcome, GuardrailEventHooks.post_call, started, operation_id)
        except Exception as error:
            logger.warning(
                "collieai guardrail post_call: the glue failed (%s); the call goes on%s",
                type(error).__name__,
                self._named,
            )
        return None

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
