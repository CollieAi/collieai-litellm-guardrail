"""The adapter in the pinned LiteLLM proxy's own code (LiteLLM 1.102.4): the loader of the config's guardrails,
a call the proxy refuses before any model, and a guardrail deleted from the registry.

These need the proxy's dependencies (FastAPI and the rest of `litellm[proxy]`); without them the module is skipped.
The proxy's HTTP server itself is not exercised here."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("litellm")
registry = pytest.importorskip("litellm.proxy.guardrails.guardrail_registry")

import litellm  # noqa: E402
from litellm.caching.dual_cache import DualCache  # noqa: E402
from litellm.proxy._types import ProxyErrorTypes, ProxyException, UserAPIKeyAuth  # noqa: E402
from litellm.proxy.guardrails.init_guardrails import init_guardrails_v2  # noqa: E402
from litellm.proxy.utils import ProxyLogging  # noqa: E402

import collieai_guardrail.observer as observer_module  # noqa: E402
from collieai_guardrail.config import ConfigError  # noqa: E402
from collieai_guardrail.guardrail import CollieAIGuardrail  # noqa: E402
from collieai_guardrail.heartbeat import interval_counts, unwired_phases  # noqa: E402
from litellm_contract import (  # noqa: E402
    API_BASE,
    API_KEY,
    MockServer,
    check_reply,
    json_reply,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore:Item 'summary' on TypedDict class 'ChatCompletionReasoningItem':UserWarning"
)

GUARDRAIL_PATH = "collieai_guardrail.guardrail.CollieAIGuardrail"
ROUTE = "/v1/chat/completions"
HANDLER = registry.IN_MEMORY_GUARDRAIL_HANDLER
Load = Callable[..., CollieAIGuardrail]


def _entry(**params: Any) -> dict[str, Any]:
    """The guardrail as the operator writes it in the proxy's config."""
    return {
        "guardrail_name": "collieai-monitor",
        "litellm_params": {
            "guardrail": GUARDRAIL_PATH,
            "mode": ["pre_call", "post_call"],
            "default_on": True,
            "api_base": API_BASE,
            "api_key": API_KEY,
            "deadline_ms": 1000,
            "network_margin_ms": 100,
            "synthetic_interval_s": 0,
            "summary_interval_s": 3600,
            **params,
        },
    }


def _ours() -> list[CollieAIGuardrail]:
    return [callback for callback in litellm.callbacks if isinstance(callback, CollieAIGuardrail)]


def _forget_ours() -> list[CollieAIGuardrail]:
    """Take every instance of the adapter out of LiteLLM's registry and callbacks, as a deletion does."""
    ours = {id(callback): callback for callback in _ours()}
    for guardrail_id, callback in list(HANDLER.guardrail_id_to_custom_guardrail.items()):
        if isinstance(callback, CollieAIGuardrail):
            HANDLER.delete_in_memory_guardrail(guardrail_id)
            ours[id(callback)] = callback
    for callback in ours.values():
        litellm.logging_callback_manager.remove_callback_from_all_lists(callback)
    return list(ours.values())


@pytest.fixture
def clean_registry() -> Iterator[None]:
    _forget_ours()
    yield
    _forget_ours()


@pytest.fixture
async def load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[Load]:
    """The guardrail through LiteLLM's own loader, `init_guardrails_v2`, as the proxy runs it at start."""
    _forget_ours()

    def make(server: MockServer, **params: Any) -> CollieAIGuardrail:
        monkeypatch.setattr(observer_module, "new_client", server.client)
        init_guardrails_v2(all_guardrails=[_entry(**params)], config_file_path=str(tmp_path / "config.yaml"))
        [guardrail] = _ours()
        return guardrail

    yield make
    for guardrail in _forget_ours():
        await guardrail.heartbeat.stop()
        await guardrail.observer.aclose()


def _server() -> MockServer:
    def respond(request: Any) -> Any:
        phase = "pre_call" if json.loads(request.content)["input_type"] == "request" else "post_call"
        return json_reply(200, check_reply(phase))(request)

    return MockServer(respond)


async def _until(condition: Callable[[], bool], what: str, *, seconds: float = 3) -> None:
    for _ in range(int(seconds * 100)):
        if condition():
            return
        await asyncio.sleep(0.01)
    raise AssertionError(what)


# ------------------------------------------------------------------ the loader


@pytest.mark.parametrize(
    ("params", "setting", "strict"),
    [
        ({"api_key": "os.environ/COLLIEAI_TEST_UNSET_KEY"}, "api_key", None),  # the registry makes it "None"
        ({"mode": "during_call"}, "mode", None),
        ({"mode": "during_call"}, "mode", "false"),  # LiteLLM itself only warns then
        ({"api_base": "https://["}, "api_base", None),  # urlsplit raises a ValueError of its own
        ({"deadline_ms": 1000, "network_margin_ms": 1000}, "network_margin_ms", None),
    ],
)
def test_litellms_loader_does_not_start_the_proxy_without_the_guardrail(
    params: dict[str, Any],
    setting: str,
    strict: str | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    clean_registry: None,
) -> None:
    """`init_guardrails_v2` logs a ValueError or a TypeError and starts without the guardrail; a ConfigError goes
    through to the proxy's start."""
    monkeypatch.delenv("COLLIEAI_TEST_UNSET_KEY", raising=False)
    if strict is None:
        monkeypatch.delenv("LITELLM_STRICT_GUARDRAIL_MODES", raising=False)
    else:
        monkeypatch.setenv("LITELLM_STRICT_GUARDRAIL_MODES", strict)
    with pytest.raises(ConfigError, match=setting) as refused:
        init_guardrails_v2(all_guardrails=[_entry(**params)], config_file_path=str(tmp_path / "config.yaml"))
    assert API_KEY not in str(refused.value)
    assert _ours() == []


async def test_litellms_loader_builds_and_registers_the_guardrail(load: Load) -> None:
    guardrail = load(_server(), forward_conversation_id=True)
    assert guardrail.guardrail_name == "collieai-monitor" and guardrail.config.forward_conversation_id is True
    assert guardrail in HANDLER.guardrail_id_to_custom_guardrail.values()


# ------------------------------------------------------------------ a call the proxy refused


async def test_a_call_the_proxy_refused_is_counted_outside_not_as_a_hook_that_did_not_run(load: Load) -> None:
    """The proxy logs a refusal before any model — a bad key here — as a failed Chat Completion with its messages,
    and its hooks never ran. It is outside, `rejected`: no eligible call unseen, no «not wired»."""
    guardrail = load(_server())
    before = guardrail.counters.snapshot()
    await ProxyLogging(user_api_key_cache=DualCache()).post_call_failure_hook(
        request_data={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hello"}]},
        original_exception=ProxyException(
            message="Invalid proxy server token passed", type="auth_error", param="None", code=401
        ),
        user_api_key_dict=UserAPIKeyAuth(request_route=ROUTE),
        error_type=ProxyErrorTypes.auth_error,
        route=ROUTE,
    )
    await _until(lambda: guardrail.counters.snapshot()["pre_call"]["logged_calls"] == 1, "the refusal was not logged")
    now = guardrail.counters.snapshot()
    counts = interval_counts(before, now)
    pre, post = counts["pre_call"], counts["post_call"]
    assert (pre["logged_calls"], pre["logged_eligible"], pre["logged_outside"]) == (1, 0, {"rejected": 1})
    assert post["logged_calls"] == 0  # a failed call had no post-call phase
    assert pre["logged_unseen"] == 0 and unwired_phases(counts, now) == []


# ------------------------------------------------------------------ what LiteLLM's spend logs keep


async def test_where_litellms_spend_log_redacts_the_response_it_keeps_the_action_and_the_categories(
    load: Load,
) -> None:
    """With LiteLLM's default (`store_prompts_in_spend_logs: false`) the spend log replaces the
    guardrail's response; what the check found stays in the fields it keeps."""
    from litellm.proxy.spend_tracking.spend_tracking_utils import _sanitize_guardrail_information_for_spend_logs

    guardrail = load(_server())
    data = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hello"}], "metadata": {}}
    await guardrail.async_pre_call_hook(
        user_api_key_dict=UserAPIKeyAuth(request_route=ROUTE), cache=None, data=data, call_type="acompletion"
    )
    [kept] = _sanitize_guardrail_information_for_spend_logs(data["metadata"]["standard_logging_guardrail_information"])
    assert kept["guardrail_response"] == "REDACTED_BY_LITELM"
    assert (kept["guardrail_status"], kept["guardrail_action"], kept["violation_categories"]) == (
        "guardrail_flagged",
        "NONE",
        ["aho_corasick"],
    )


# ------------------------------------------------------------------ a deleted guardrail


async def test_a_deleted_guardrail_stops_its_heartbeat_and_closes_its_client(load: Load) -> None:
    server = _server()
    guardrail = load(server, synthetic_interval_s=1, summary_interval_s=1, deadline_ms=200)
    await _until(lambda: len(server.requests) == 2, "the synthetic check at start did not run")
    task = guardrail.heartbeat._task
    assert task is not None
    [guardrail_id] = [key for key, value in HANDLER.guardrail_id_to_custom_guardrail.items() if value is guardrail]
    HANDLER.delete_in_memory_guardrail(guardrail_id)
    await asyncio.wait_for(task, timeout=3)  # the next turn sees the removal, then waits a deadline and closes
    assert guardrail.observer._client.is_closed
    assert len(server.requests) == 2  # no synthetic check with the deleted guardrail's key
    guardrail.heartbeat.ensure_running()  # a hook already dispatched to it does not bring it back
    assert guardrail.heartbeat._task is task and task.done()


async def test_a_guardrail_deleted_before_its_heartbeat_ever_ran_stops_too(load: Load) -> None:
    """Loaded and at once deleted, with no turn of the event loop in between: the heartbeat never saw it among
    LiteLLM's callbacks, and still sends nothing with the deleted guardrail's key."""
    server = _server()
    guardrail = load(server, synthetic_interval_s=1, summary_interval_s=1, deadline_ms=200)
    [guardrail_id] = [key for key, value in HANDLER.guardrail_id_to_custom_guardrail.items() if value is guardrail]
    HANDLER.delete_in_memory_guardrail(guardrail_id)
    task = guardrail.heartbeat._task
    assert task is not None
    await asyncio.wait_for(task, timeout=3)
    assert guardrail.observer._client.is_closed
    assert server.requests == []
