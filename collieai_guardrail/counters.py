"""The adapter's in-process counts, with a snapshot.

Per phase, from the hooks:

- `hook_calls` — the hook ran for a call; `outside` by reason — the call is outside the supported profile, nothing
  was sent;
- `eligible` — the call is in the supported profile; each one ends as exactly one of `completed` (a check) or a
  `bypass` by reason; `attempted` — the transport started the POST: a call over the body limit, or one whose deadline
  was gone before the POST could start (`timeout`, `no_time_left`), is a bypass with nothing sent.

And from a path apart from the hooks — LiteLLM's success and failure log events, which run whether or not a hook
did — `logged_calls` and `logged_eligible`, one per call however many attempts LiteLLM logged for it, and
`logged_repeats`, the further attempts. A hook that never ran shows as `logged_eligible` above `eligible`, in a
count its own failure cannot hide. Counts only: no text, no identifier, nothing per call.
"""

from __future__ import annotations

import threading
from collections import Counter

PHASES = ("pre_call", "post_call")
TOTALS = ("hook_calls", "eligible", "attempted", "completed", "logged_calls", "logged_eligible", "logged_repeats")


class Counters:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._totals: Counter[tuple[str, str]] = Counter()
        self._by_reason: Counter[tuple[str, str, str]] = Counter()

    def add(self, phase: str, name: str) -> None:
        if phase not in PHASES or name not in TOTALS:
            raise ValueError(f"unknown count {phase}/{name}")
        with self._lock:
            self._totals[(phase, name)] += 1

    def add_reason(self, phase: str, kind: str, reason: str) -> None:
        """`kind` is `bypass`, `outside` or `logged_outside`."""
        if phase not in PHASES or kind not in ("bypass", "outside", "logged_outside"):
            raise ValueError(f"unknown count {phase}/{kind}")
        with self._lock:
            self._by_reason[(phase, kind, reason)] += 1

    def snapshot(self) -> dict[str, dict[str, object]]:
        with self._lock:
            totals = dict(self._totals)
            by_reason = dict(self._by_reason)
        out: dict[str, dict[str, object]] = {}
        for phase in PHASES:
            counts: dict[str, object] = {name: totals.get((phase, name), 0) for name in TOTALS}
            for kind in ("bypass", "outside", "logged_outside"):
                counts[kind] = {
                    reason: n for (p, k, reason), n in sorted(by_reason.items()) if p == phase and k == kind
                }
            out[phase] = counts
        return out
