"""The worker's own evidence that the adapter is wired and the server answers.

- **Summary.** Every `summary_interval_s` one INFO line, `collieai guardrail summary {json}`, with the guardrail's
  name in LiteLLM (`guardrail`: a proxy may have more than one CollieAi guardrail) and each phase's counts over the
  interval: hooks, eligible, attempted, completed, bypass and outside by reason, and what LiteLLM's
  log events counted apart from the hooks. Derived: `coverage` — of the checks that ended in the interval, the
  share that completed, `completed / (completed + bypass)`; a check started in one interval may end in the next,
  so it is not taken of `eligible`, which counts starts. `outside_share` — calls outside the supported profile of all
  the hook saw, so a large excluded share does not hide behind a good coverage. `logged_unseen` — eligible calls LiteLLM
  logged less those a hook saw in the same interval: a call that crosses the interval's end moves it by one, and a
  call the hook saw that the proxy then refused (another guardrail's block) takes one off.
- **Not wired.** A phase whose hook has never run in this instance, while LiteLLM logged eligible calls in the
  interval, gets a WARNING: the guardrail is not wired for it. Never, not in the interval: a hook that ran in one
  interval for a call LiteLLM logs in the next is wired.
- **Synthetic check.** At start — within SYNTHETIC_START_SPREAD_S, at random, so workers that start together seldom
  meet at the server — and every `synthetic_interval_s`, one pre-call and one post-call observation of a fixed
  text — never a user's — through the same client, key and deadline. It is not counted with the calls' traffic.
  Both completed: INFO. Answered `busy` and nothing worse: INFO, and the check is repeated soon, up to
  SYNTHETIC_BUSY_RETRIES times. Otherwise, or busy every time: a WARNING with each phase's result and reason.

Counts, codes, results and the guardrail's name only: no text, no key, no identifier of a call. The loop survives its
own errors; cancelling the task stops it, and so does the owner's retiring: the loop asks at each turn, and a retired
owner's loop ends for good.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import random
import uuid
from collections.abc import Awaitable, Callable, Mapping

from collieai_guardrail.config import named
from collieai_guardrail.counters import PHASES, Counters
from collieai_guardrail.observer import POST_CALL, PRE_CALL, Observer, Outcome
from collieai_guardrail.payload import CallFacts

logger = logging.getLogger("collieai_guardrail")

#: The synthetic check's call: a fixed text that carries nothing of a user's.
SYNTHETIC_TEXT = "CollieAi synthetic wiring check"
SYNTHETIC_REQUEST: dict[str, object] = {
    "model": "collieai-synthetic",
    "messages": [{"role": "user", "content": SYNTHETIC_TEXT}],
}
#: The first synthetic check waits a random part of this, at most a tenth of its interval. Workers that start together
#: — a proxy's, or several proxies' — would send theirs at one moment, and a server that admits one check at a time
#: per process answers some of them `busy`: a false alarm at start (seen with two proxy workers).
SYNTHETIC_START_SPREAD_S = 3.0
#: The spread makes a meeting rare, not impossible. A synthetic check answered `busy` — the server is up, took the key
#: and had no free slot — is repeated after a random wait between half the spread and the whole of it, up to this many
#: times; busy every time is a WARNING: no free slot for several seconds is the server's capacity, not a meeting.
SYNTHETIC_BUSY_RETRIES = 2
SYNTHETIC_RESPONSE: dict[str, object] = {
    "object": "chat.completion",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": SYNTHETIC_TEXT}}],
}


def interval_counts(
    before: Mapping[str, Mapping[str, object]], after: Mapping[str, Mapping[str, object]]
) -> dict[str, dict[str, object]]:
    """Each phase's counts between two snapshots of `Counters`, with the derived shares."""
    out: dict[str, dict[str, object]] = {}
    for phase in PHASES:
        then, now = before.get(phase, {}), after.get(phase, {})
        counts: dict[str, object] = {}
        for name, value in now.items():
            if isinstance(value, int):
                counts[name] = value - _int(then.get(name))
            elif isinstance(value, Mapping):
                kept = then.get(name)
                previous: Mapping[str, object] = kept if isinstance(kept, Mapping) else {}
                counts[name] = {
                    reason: n - _int(previous.get(reason))
                    for reason, n in value.items()
                    if isinstance(n, int) and n - _int(previous.get(reason))
                }
        completed = _int(counts.get("completed"))
        counts["coverage"] = _share(completed, completed + _total(counts.get("bypass")))
        counts["outside_share"] = _share(_total(counts.get("outside")), _int(counts.get("hook_calls")))
        counts["logged_unseen"] = _int(counts.get("logged_eligible")) - _int(counts.get("eligible"))
        out[phase] = counts
    return out


def unwired_phases(counts: Mapping[str, Mapping[str, object]], totals: Mapping[str, Mapping[str, object]]) -> list[str]:
    """Phases whose hook has never run — `totals`, the instance's counts so far — while LiteLLM logged eligible calls
    in the interval (`counts`)."""
    return [
        phase
        for phase in PHASES
        if _int(counts.get(phase, {}).get("logged_eligible")) > 0 and _int(totals.get(phase, {}).get("hook_calls")) == 0
    ]


async def synthetic_check(observer: Observer) -> tuple[Outcome, Outcome]:
    """One pre-call and one post-call observation of the fixed text, not counted with the calls."""
    facts = CallFacts(operation_id=f"synthetic-{uuid.uuid4().hex}", model="collieai-synthetic")
    pre = await observer.observe_request(SYNTHETIC_REQUEST, call_type="acompletion", facts=facts, count=False)
    post = await observer.observe_response(
        SYNTHETIC_REQUEST, SYNTHETIC_RESPONSE, call_type="acompletion", facts=facts, count=False
    )
    return pre, post


def only_busy(outcomes: tuple[Outcome, ...]) -> bool:
    """Some phase was answered `busy` and every other completed."""
    busy = [outcome.result == "bypass" and outcome.reason == "busy" for outcome in outcomes]
    return any(busy) and all(b or outcome.result == "completed" for b, outcome in zip(busy, outcomes, strict=True))


def describe(outcome: Outcome) -> str:
    """A closed code for an outcome: `completed`, `bypass:<reason>` or `outside:<reason>`."""
    return outcome.result if outcome.reason is None else f"{outcome.result}:{outcome.reason}"


class Heartbeat:
    def __init__(
        self,
        observer: Observer,
        counters: Counters,
        *,
        summary_interval_s: int,
        synthetic_interval_s: int,
        retired: Callable[[], bool] | None = None,
        on_retired: Callable[[], Awaitable[None]] | None = None,
        start_spread_s: float | None = None,
        name: str | None = None,
    ):
        """`retired` — asked at each turn of the loop: the owner is gone and the loop should end; then
        `on_retired` releases what the owner holds. `start_spread_s` — the first synthetic check waits a random part
        of it; by default SYNTHETIC_START_SPREAD_S, at most a tenth of the interval. `name` — the guardrail's name in
        LiteLLM, in each line."""
        self._name = name
        self._named = named(name)
        self._observer = observer
        self._counters = counters
        self._summary_interval_s = summary_interval_s
        self._synthetic_interval_s = synthetic_interval_s
        self._start_spread_s = (
            min(SYNTHETIC_START_SPREAD_S, synthetic_interval_s / 10) if start_spread_s is None else start_spread_s
        )
        self._retired = retired
        self._on_retired = on_retired
        self._last = counters.snapshot()
        self._synthetic: dict[str, str] = {}
        self._task: asyncio.Task[None] | None = None
        self._ended = False

    def ensure_running(self) -> None:
        """Start the loop on the running event loop unless it runs there already or has ended for good; outside a
        loop, do nothing."""
        if self._ended:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        task = self._task
        if task is not None and not task.done() and task.get_loop() is loop:
            return
        self._task = loop.create_task(self._run(), name="collieai-guardrail-heartbeat")

    async def stop(self) -> None:
        """End the loop for good."""
        self._ended = True
        task, self._task = self._task, None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        next_synthetic = (
            loop.time() + random.uniform(0, self._start_spread_s) if self._synthetic_interval_s else math.inf
        )
        next_summary = loop.time() + self._summary_interval_s
        busy_retries = 0
        while True:
            if self._retire_if_gone():
                if self._on_retired is not None:
                    await self._on_retired()
                return
            if loop.time() >= next_synthetic:
                may_retry = busy_retries < SYNTHETIC_BUSY_RETRIES
                outcomes = await self.run_synthetic(retry_if_busy=may_retry)
                if may_retry and outcomes is not None and only_busy(outcomes):
                    busy_retries += 1
                    next_synthetic = loop.time() + random.uniform(self._start_spread_s / 2, self._start_spread_s)
                else:
                    busy_retries = 0
                    next_synthetic = loop.time() + self._synthetic_interval_s
            if loop.time() >= next_summary:
                self.summarize()
                next_summary = loop.time() + self._summary_interval_s
            await asyncio.sleep(max(0.0, min(next_synthetic, next_summary) - loop.time()))

    def _retire_if_gone(self) -> bool:
        try:
            gone = self._retired is not None and self._retired()
        except Exception as error:  # an owner that cannot say is kept: a stopped heartbeat would hide it
            logger.warning(
                "collieai guardrail: checking the registration failed (%s)%s", type(error).__name__, self._named
            )
            return False
        if gone:
            self._ended = True
            logger.info(
                "collieai guardrail: LiteLLM no longer has this instance among its callbacks (the guardrail was"
                " removed or rebuilt); its summary and synthetic check stop%s",
                self._named,
            )
        return gone

    async def run_synthetic(self, *, retry_if_busy: bool = False) -> tuple[Outcome, Outcome] | None:
        """One synthetic check, logged. With `retry_if_busy`, a check answered `busy` and nothing worse is INFO: the
        loop repeats it."""
        try:
            pre, post = await synthetic_check(self._observer)
        except Exception as error:  # the loop outlives its own faults
            logger.warning(
                "collieai guardrail synthetic check failed inside the adapter (%s)%s", type(error).__name__, self._named
            )
            self._synthetic = {PRE_CALL: "error", POST_CALL: "error"}
            self._report_refusing()
            return None
        self._synthetic = {PRE_CALL: describe(pre), POST_CALL: describe(post)}
        if pre.result == post.result == "completed":
            logger.info("collieai guardrail synthetic check: pre_call completed, post_call completed%s", self._named)
        elif retry_if_busy and only_busy((pre, post)):
            logger.info(
                "collieai guardrail synthetic check: pre_call %s, post_call %s; CollieAi had no free slot, the check is"
                " repeated in a few seconds%s",
                describe(pre),
                describe(post),
                self._named,
            )
        else:
            logger.warning(
                "collieai guardrail synthetic check: pre_call %s (detail=%s, http_status=%s, rule_id=%s), post_call %s"
                " (detail=%s, http_status=%s, rule_id=%s); the calls' observations are likely failing the same way%s",
                describe(pre),
                pre.detail,
                pre.http_status,
                pre.rule_id,
                describe(post),
                post.detail,
                post.http_status,
                post.rule_id,
                self._named,
            )
            self._report_refusing()
        return pre, post

    def _report_refusing(self) -> None:
        """In block a failed synthetic check means the calls' checks are likely failing the same way, and each such
        call is refused: an ERROR, whatever failed."""
        if self._observer.config.enforce:
            logger.error(
                "collieai guardrail synthetic check failed in enforcement_mode block: calls are likely being"
                " refused (503 guardrail_unavailable)%s",
                self._named,
            )

    def summarize(self) -> dict[str, object] | None:
        try:
            now = self._counters.snapshot()
            counts = interval_counts(self._last, now)
            self._last = now
            summary: dict[str, object] = {
                "guardrail": self._name,
                "interval_s": self._summary_interval_s,
                **counts,
                "synthetic": dict(self._synthetic),
            }
            logger.info("collieai guardrail summary %s", json.dumps(summary, sort_keys=True, separators=(",", ":")))
            for phase in unwired_phases(counts, now):
                logger.warning(
                    "collieai guardrail %s: LiteLLM logged %d eligible calls and the hook has never run; the guardrail"
                    " is not wired for this phase (mode, default_on or the request's guardrails)%s",
                    phase,
                    _int(counts[phase].get("logged_eligible")),
                    self._named,
                )
            return summary
        except Exception as error:
            logger.warning(
                "collieai guardrail summary failed inside the adapter (%s)%s", type(error).__name__, self._named
            )
            return None


def _int(value: object) -> int:
    return value if type(value) is int else 0


def _total(by_reason: object) -> int:
    return sum(_int(n) for n in by_reason.values()) if isinstance(by_reason, Mapping) else 0


def _share(part: int, whole: int) -> float | None:
    return round(part / whole, 4) if whole > 0 else None
