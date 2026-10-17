"""One phase of one call, end to end.

`observe_request` (pre-call) and `observe_response` (post-call) decide eligibility, build and send the one request,
and read the reply under the phase's deadline, which starts when the phase does. They never raise — only a
cancellation of the caller goes on, counted first as the `cancelled` bypass — and never change what they are given:
the outcome is a value the LiteLLM glue records. Every outcome is exactly one of a completed check, a bypass by
reason, or outside the supported profile by reason. Whatever a reply says, it is read by the deadline or it is
`timeout`.

Logs carry the phase, the reason, a fixed detail code and the HTTP status — never a text, a key or an identifier
of the user's call. A bypass of a call is logged at WARNING at most once a minute per phase and reason, with the
number held back since; the counters have every one. A synthetic check's (`count=False`) is not: the heartbeat
reports its outcome, and it takes no place of a call's warning.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Literal

import httpx

from collieai_guardrail import __version__
from collieai_guardrail.config import MAX_REPLY_BYTES, AdapterConfig
from collieai_guardrail.counters import Counters
from collieai_guardrail.eligibility import request_outside_reason, response_text
from collieai_guardrail.payload import CallFacts, Sender, encode, post_call_payload, pre_call_payload
from collieai_guardrail.transport import Failure, new_client, post_once
from collieai_guardrail.verdict import Check, Rejected, error_reason, parse_check

logger = logging.getLogger("collieai_guardrail")

PRE_CALL = "pre_call"
POST_CALL = "post_call"
#: LiteLLM's `GuardrailStatus` for each result. A check that found something is `guardrail_flagged` — LiteLLM's
#: «allowed content through but recorded a non-blocking violation»: Monitor never intervenes, so never
#: `guardrail_intervened`. A bypass is never `success`.
GuardrailStatus = Literal["success", "guardrail_flagged", "guardrail_failed_to_respond", "not_run"]
GUARDRAIL_STATUS: dict[str, GuardrailStatus] = {
    "completed": "success",
    "bypass": "guardrail_failed_to_respond",
    "outside": "not_run",
}
FLAGGED_STATUS: GuardrailStatus = "guardrail_flagged"
WARNING_INTERVAL_S = 60.0


@dataclass(frozen=True)
class Outcome:
    phase: str
    result: str  # completed | bypass | outside
    reason: str | None = None
    detail: str | None = None
    http_status: int | None = None
    check: Check | None = None
    duration_s: float = 0.0

    @property
    def guardrail_status(self) -> GuardrailStatus:
        if self.check is not None and self.check.flagged:
            return FLAGGED_STATUS
        return GUARDRAIL_STATUS[self.result]

    def diagnostics(self) -> dict[str, object]:
        """For LiteLLM's standard guardrail log: no text. What the check found (`would`), what happened to the call
        (`effective` — always `allow`: Monitor changes nothing) and whether a detector degraded are separate."""
        out: dict[str, object] = {
            "adapter": "collieai-native",
            "phase": self.phase,
            "result": self.result,
            "effective": "allow",
        }
        if self.check is not None:
            out["would"] = self.check.would_outcome
            out["degraded"] = self.check.coverage_state == "degraded"
            out["collieai"] = self.check.diagnostics
        else:
            out["reason"] = self.reason
            if self.detail is not None:
                out["detail"] = self.detail
            if self.http_status is not None:
                out["http_status"] = self.http_status
        return out


class Observer:
    def __init__(
        self,
        config: AdapterConfig,
        *,
        litellm_version: str | None = None,
        client: httpx.AsyncClient | None = None,
        counters: Counters | None = None,
    ) -> None:
        self.config = config
        self.counters = counters if counters is not None else Counters()
        self._sender = Sender(adapter_version=__version__, litellm_version=litellm_version, budget_ms=config.budget_ms)
        # made here, not on the first call: two first calls at once would otherwise make two. httpx binds a client
        # to no event loop until it connects.
        self._client = client if client is not None else new_client()
        self._headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": f"collieai-litellm-guardrail/{__version__}",
        }
        self._warned: dict[tuple[str, str], tuple[float, int]] = {}

    async def aclose(self) -> None:
        await self._client.aclose()

    async def observe_request(
        self, request: Mapping[str, object], *, call_type: str | None, facts: CallFacts, count: bool = True
    ) -> Outcome:
        """`call_type` — LiteLLM's type of the call; None, not known, is outside the supported profile."""
        started = asyncio.get_running_loop().time()
        tally = self.counters if count else Counters()
        tally.add(PRE_CALL, "hook_calls")
        try:
            reason = request_outside_reason(request, call_type)
        except Exception:
            reason = "malformed"
        if reason is not None:
            return self._outside(tally, PRE_CALL, reason, started)
        tally.add(PRE_CALL, "eligible")
        try:
            payload = pre_call_payload(request, self._sent_facts(facts), self._sender)
            return await self._exchange(tally, PRE_CALL, payload, started)
        except asyncio.CancelledError:
            self._cancelled(tally, PRE_CALL)
            raise
        except Exception as error:  # nothing else escapes the hook
            return self._bypass(tally, PRE_CALL, "adapter_error", started, detail=type(error).__name__)

    async def observe_response(
        self,
        request: Mapping[str, object],
        response: object,
        *,
        call_type: str | None,
        facts: CallFacts,
        count: bool = True,
    ) -> Outcome:
        """`call_type` — the same call's type as pre-call had it: the request alone does not say whether the call
        was a Chat Completion."""
        started = asyncio.get_running_loop().time()
        tally = self.counters if count else Counters()
        tally.add(POST_CALL, "hook_calls")
        text: str | None = None
        try:
            reason = request_outside_reason(request, call_type)
            if reason is None:
                text, reason = response_text(response)
        except Exception:
            reason = "malformed"
        if reason is not None or text is None:
            return self._outside(tally, POST_CALL, reason or "malformed", started)
        tally.add(POST_CALL, "eligible")
        try:
            payload = post_call_payload(request, text, self._sent_facts(facts), self._sender)
            return await self._exchange(tally, POST_CALL, payload, started)
        except asyncio.CancelledError:
            self._cancelled(tally, POST_CALL)
            raise
        except Exception as error:
            return self._bypass(tally, POST_CALL, "adapter_error", started, detail=type(error).__name__)

    def _sent_facts(self, facts: CallFacts) -> CallFacts:
        """The facts as they may leave: the conversation ID only when the operator turned its forwarding on,
        whatever the caller of the observer put in."""
        return facts if self.config.forward_conversation_id else replace(facts, conversation_id=None)

    async def _exchange(self, tally: Counters, phase: str, payload: dict[str, object], started: float) -> Outcome:
        loop = asyncio.get_running_loop()
        deadline = started + self.config.deadline_ms / 1000
        body = encode(payload)
        if len(body) > self.config.max_body_bytes:
            return self._bypass(tally, phase, "limit", started, detail="body_over_limit")
        sent = await post_once(
            self._client,
            url=self.config.url,
            headers=self._headers,
            body=body,
            deadline=deadline,
            max_reply_bytes=MAX_REPLY_BYTES,
            on_attempt=lambda: tally.add(phase, "attempted"),  # by the transport: the POST started
        )
        if isinstance(sent, Failure):
            return self._bypass(tally, phase, sent.reason, started, detail=sent.detail)
        parsed: Check | Rejected | None = None
        reason: str = "protocol"
        detail: str | None = None
        if sent.status == 200:
            parsed = parse_check(sent.body, phase=phase)
            detail = parsed.detail if isinstance(parsed, Rejected) else None
        elif 200 <= sent.status < 300:
            detail = "status"
        else:
            reason = error_reason(sent.status, sent.body)
        # a reply read after the deadline is no answer, whatever it says: a late result is not a check
        if loop.time() > deadline:
            return self._bypass(tally, phase, "timeout", started, detail="late", http_status=sent.status)
        if not isinstance(parsed, Check):
            return self._bypass(tally, phase, reason, started, detail=detail, http_status=sent.status)
        tally.add(phase, "completed")
        logger.debug("collieai guardrail %s: completed, coverage %s", phase, parsed.coverage_state)
        return Outcome(phase, "completed", check=parsed, http_status=200, duration_s=loop.time() - started)

    def _cancelled(self, tally: Counters, phase: str) -> None:
        # the caller's own cancellation, not a fault of the adapter or the server: counted, never warned about
        tally.add_reason(phase, "bypass", "cancelled")
        logger.debug("collieai guardrail %s: the call was cancelled during the check", phase)

    def _outside(self, tally: Counters, phase: str, reason: str, started: float) -> Outcome:
        tally.add_reason(phase, "outside", reason)
        logger.debug("collieai guardrail %s: outside the supported profile (%s), not sent", phase, reason)
        return Outcome(phase, "outside", reason=reason, duration_s=asyncio.get_running_loop().time() - started)

    def _bypass(
        self,
        tally: Counters,
        phase: str,
        reason: str,
        started: float,
        *,
        detail: str | None = None,
        http_status: int | None = None,
    ) -> Outcome:
        tally.add_reason(phase, "bypass", reason)
        if tally is self.counters:  # a call's; a synthetic check counts apart, and the heartbeat reports it
            self._warn(phase, reason, detail, http_status)
        return Outcome(
            phase,
            "bypass",
            reason=reason,
            detail=detail,
            http_status=http_status,
            duration_s=asyncio.get_running_loop().time() - started,
        )

    def _warn(self, phase: str, reason: str, detail: str | None, http_status: int | None) -> None:
        now = time.monotonic()
        last, held = self._warned.get((phase, reason), (float("-inf"), 0))
        if now - last < WARNING_INTERVAL_S:
            self._warned[(phase, reason)] = (last, held + 1)
            return
        self._warned[(phase, reason)] = (now, 0)
        logger.warning(
            "collieai guardrail %s: observation bypassed (%s, detail=%s, http_status=%s); the call goes on unchanged"
            " (%d more held back since the last warning)",
            phase,
            reason,
            detail,
            http_status,
            held,
        )
