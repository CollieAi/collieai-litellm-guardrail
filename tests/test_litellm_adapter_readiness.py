"""The readiness guard (`RequireGuardrail`): LiteLLM starts without a guardrail whose entry it refuses, so block mode
needs one more line, the guard in `litellm_settings.callbacks`. It refuses every call while the worker has no CollieAi
guardrail in block, and the guardrail in block does not start without it. Through LiteLLM's own loaders and dispatch."""

from __future__ import annotations

import copy
import importlib.util
import logging
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("litellm")
registry = pytest.importorskip("litellm.proxy.guardrails.guardrail_registry")

import litellm  # noqa: E402
from litellm.caching.dual_cache import DualCache  # noqa: E402
from litellm.exceptions import GuardrailRaisedException  # noqa: E402
from litellm.integrations.custom_guardrail import CustomGuardrail  # noqa: E402
from litellm.integrations.custom_logger import CustomLogger  # noqa: E402
from litellm.proxy._types import UserAPIKeyAuth  # noqa: E402
from litellm.proxy.common_utils.callback_utils import initialize_callbacks_on_proxy  # noqa: E402
from litellm.proxy.guardrails.init_guardrails import init_guardrails_v2  # noqa: E402
from litellm.proxy.utils import ProxyLogging  # noqa: E402

import collieai_guardrail.guardrail as guardrail_module  # noqa: E402
import collieai_guardrail.observer as observer_module  # noqa: E402
from collieai_guardrail.config import ConfigError  # noqa: E402
from collieai_guardrail.guardrail import (  # noqa: E402
    GUARD_LOG_INTERVAL_S,
    GUARD_PATH,
    CollieAIGuardrail,
    RequireGuardrail,
    require_guardrail,
)
from litellm_contract import (  # noqa: E402
    API_BASE,
    API_KEY,
    MockServer,
    chat_request,
    check_reply,
    enforce_reply,
    json_reply,
)

HANDLER = registry.IN_MEMORY_GUARDRAIL_HANDLER
USER = UserAPIKeyAuth(request_route="/v1/chat/completions")


def _entry(**params: Any) -> dict[str, Any]:
    litellm_params = {
        "guardrail": "collieai_guardrail.guardrail.CollieAIGuardrail",
        "mode": ["pre_call", "post_call"],
        "default_on": True,
        "api_base": API_BASE,
        "api_key": API_KEY,
        "enforcement_mode": "block",
        "synthetic_interval_s": 0,
        "summary_interval_s": 3600,
        **params,
    }
    return {"guardrail_name": "collieai-block", "litellm_params": {k: v for k, v in litellm_params.items() if v != ...}}


def _forget_ours() -> list[CollieAIGuardrail]:
    ours = {id(c): c for c in litellm.callbacks if isinstance(c, CollieAIGuardrail)}
    for guardrail_id, callback in list(HANDLER.guardrail_id_to_custom_guardrail.items()):
        if isinstance(callback, CollieAIGuardrail):
            HANDLER.delete_in_memory_guardrail(guardrail_id)
            ours[id(callback)] = callback
    for callback in ours.values():
        litellm.logging_callback_manager.remove_callback_from_all_lists(callback)
    return list(ours.values())


@pytest.fixture
async def proxy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[Any]:
    """A worker's callbacks as the proxy loads them: the guard first, from its path, then the guardrail entries."""
    _forget_ours()
    monkeypatch.setattr(litellm, "callbacks", [])
    monkeypatch.setattr(require_guardrail, "_reported_at", None)
    config = str(tmp_path / "config.yaml")
    initialize_callbacks_on_proxy(value=[GUARD_PATH], premium_user=False, config_file_path=config, litellm_settings={})

    def load(server: MockServer, **params: Any) -> list[CollieAIGuardrail]:
        monkeypatch.setattr(observer_module, "new_client", server.client)
        init_guardrails_v2(all_guardrails=[_entry(**params)], config_file_path=config)
        return [c for c in litellm.callbacks if isinstance(c, CollieAIGuardrail)]

    yield load
    for guardrail in _forget_ours():
        await guardrail.heartbeat.stop()
        await guardrail.observer.aclose()


async def _dispatch(data: dict[str, Any] | None = None) -> dict[str, Any]:
    data = copy.deepcopy(chat_request()) if data is None else data
    await ProxyLogging(user_api_key_cache=DualCache()).pre_call_hook(
        user_api_key_dict=USER, data=data, call_type="acompletion"
    )
    return data


# ------------------------------------------------------------------ the pairing


def test_litellms_callbacks_loader_takes_the_guard_by_its_path(proxy: Any) -> None:
    assert [c for c in litellm.callbacks if isinstance(c, RequireGuardrail)] == [require_guardrail]


def test_block_does_not_start_without_the_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the guard a block entry LiteLLM refuses would leave every call unchecked; the error says what to add."""
    monkeypatch.setattr(litellm, "callbacks", [])
    with pytest.raises(ConfigError, match=f'needs the readiness guard: add "{GUARD_PATH}" to litellm_settings'):
        CollieAIGuardrail(
            api_base=API_BASE,
            api_key=API_KEY,
            enforcement_mode="block",
            event_hook=["pre_call", "post_call"],
            default_on=True,
        )


async def test_monitor_needs_no_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "callbacks", [])
    guardrail = CollieAIGuardrail(
        api_base=API_BASE,
        api_key=API_KEY,
        event_hook=["pre_call", "post_call"],
        default_on=True,
        synthetic_interval_s=0,
    )
    await guardrail.heartbeat.stop()
    await guardrail.observer.aclose()


# ------------------------------------------------------------------ the guard in LiteLLM's dispatch


@pytest.mark.parametrize(
    "params",
    [{"mode": ...}, {"default_on": "maybe"}, {"mode": 5}],
    ids=["mode missing", "default_on: maybe", "mode: 5"],
)
async def test_an_entry_litellm_refuses_leaves_every_call_refused(proxy: Any, params: dict[str, Any]) -> None:
    """LiteLLM skips the entry ("Skipping guardrail") and the proxy would serve the call unchecked; the guard
    refuses it instead, as a check not made."""
    assert proxy(MockServer(json_reply(200, check_reply())), **params) == []

    with pytest.raises(GuardrailRaisedException) as caught:
        await _dispatch()

    refusal = caught.value
    assert (refusal.status_code, refusal.message, refusal.guardrail_name, refusal.blocked_content) == (
        503,
        "CollieAi guardrail not loaded (guardrail_not_loaded)",
        "collieai-readiness",
        False,  # a batch upload is then refused whole, not its records dropped as blocked content
    )


async def test_the_guard_refuses_in_litellms_guardrail_only_scans(proxy: Any) -> None:
    """A batch file's records and the MCP tool catalog are scanned by guardrails only; the guard runs there too."""
    with pytest.raises(GuardrailRaisedException) as caught:
        await ProxyLogging(user_api_key_cache=DualCache()).pre_call_hook(
            user_api_key_dict=USER, data=copy.deepcopy(chat_request()), call_type="acompletion", guardrails_only=True
        )
    assert caught.value.status_code == 503


async def test_a_block_guardrail_loaded_lets_the_call_through_to_it(proxy: Any) -> None:
    [guardrail] = proxy(MockServer(json_reply(200, enforce_reply())))

    data = await _dispatch()

    [record] = data["metadata"]["standard_logging_guardrail_information"]
    assert (record["guardrail_provider"], record["guardrail_status"]) == ("collieai", "success")
    assert guardrail.counters.snapshot()["pre_call"]["completed"] == 1


@pytest.mark.parametrize(
    "params",
    [{"enforcement_mode": "monitor"}, {"enforcement_mode": ..., "enforcment_mode": "block"}],
    ids=["monitor", "enforcement_mode misspelled"],
)
async def test_a_guardrail_in_monitor_is_not_enough(proxy: Any, params: dict[str, Any]) -> None:
    """A misspelled key reaches the adapter as an unknown setting and leaves it in Monitor: the guard refuses rather
    than let the proxy observe quietly. Going back to Monitor takes the guard's line out too."""
    [guardrail] = proxy(MockServer(json_reply(200, check_reply())), **params)
    assert guardrail.enforce is False

    with pytest.raises(GuardrailRaisedException) as caught:
        await _dispatch()

    assert caught.value.status_code == 503
    assert guardrail.counters.snapshot()["pre_call"]["hook_calls"] == 0


async def test_the_guard_refuses_once_the_guardrail_is_deleted(proxy: Any) -> None:
    """A guardrail deleted from LiteLLM leaves the worker without one: the guard refuses, it does not let calls go."""
    proxy(MockServer(json_reply(200, enforce_reply())))
    for guardrail in _forget_ours():
        await guardrail.heartbeat.stop()
        await guardrail.observer.aclose()

    with pytest.raises(GuardrailRaisedException):
        await _dispatch()


async def test_the_guards_error_line_comes_at_most_once_a_minute(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(litellm, "callbacks", [])
    guard = RequireGuardrail()

    async def refused() -> None:
        with pytest.raises(GuardrailRaisedException):
            await guard.async_pre_call_hook(USER, DualCache(), copy.deepcopy(chat_request()), "acompletion")

    def lines() -> list[str]:
        return [r.getMessage() for r in caplog.records if r.name == "collieai_guardrail"]

    with caplog.at_level(logging.ERROR, logger="collieai_guardrail"):
        await refused()
        assert len(lines()) == 1
        await refused()  # within the minute: refused, not logged again
        assert len(lines()) == 1
        assert guard._reported_at is not None
        guard._reported_at -= GUARD_LOG_INTERVAL_S + 1
        await refused()
        assert len(lines()) == 2

    assert all("in enforcement_mode block" in line and "(503 guardrail_not_loaded)" in line for line in lines())
    assert all(line.endswith("(guardrail collieai-readiness)") for line in lines())


def test_litellm_runs_the_guards_own_hook() -> None:
    """LiteLLM calls a plain callback's pre-call hook only when its class defines one (1.106) and, in a guardrail-only
    scan, only when it judges the request."""
    assert "async_pre_call_hook" in vars(RequireGuardrail)
    assert RequireGuardrail.enforces_request_content is True


# ------------------------------------------------------------------ what counts as the guard and as the guardrail


def _block(**params: Any) -> CollieAIGuardrail:
    return CollieAIGuardrail(
        api_base=API_BASE,
        api_key=API_KEY,
        enforcement_mode="block",
        event_hook=["pre_call", "post_call"],
        default_on=True,
        synthetic_interval_s=0,
        **params,
    )


async def _close(*guardrails: CollieAIGuardrail) -> None:
    for guardrail in guardrails:
        await guardrail.heartbeat.stop()
        await guardrail.observer.aclose()


async def _guard_lets_through(guard: Any) -> bool:
    try:
        await guard.async_pre_call_hook(USER, DualCache(), copy.deepcopy(chat_request()), "acompletion")
    except GuardrailRaisedException:
        return False
    return True


def test_another_callback_is_not_the_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "callbacks", [CustomLogger()])
    with pytest.raises(ConfigError, match="needs the readiness guard"):
        _block()


async def test_another_vendors_guardrail_in_block_is_not_enough(monkeypatch: pytest.MonkeyPatch) -> None:
    """The CollieAi entry skipped, the proxy would serve with someone else's guardrail only."""

    class Elsewhere(CustomGuardrail):
        enforce = True

    monkeypatch.setattr(litellm, "callbacks", [Elsewhere(guardrail_name="elsewhere")])
    assert await _guard_lets_through(RequireGuardrail()) is False


async def test_one_block_guardrail_among_several_is_enough(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "callbacks", [require_guardrail])
    monitor = CollieAIGuardrail(
        api_base=API_BASE,
        api_key=API_KEY,
        event_hook=["pre_call", "post_call"],
        default_on=True,
        synthetic_interval_s=0,
    )
    block = _block()
    monkeypatch.setattr(litellm, "callbacks", [require_guardrail, monitor, block])
    try:
        assert await _guard_lets_through(require_guardrail) is True
    finally:
        await _close(monitor, block)


async def test_a_second_copy_of_the_module_still_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    """LiteLLM loads a module that sits next to its config by its file path, so the guard and the guardrail may come
    from two copies of this module; a correct block config still starts and still lets calls through."""
    spec = importlib.util.spec_from_file_location("collieai_guardrail_copy", guardrail_module.__file__)
    assert spec is not None and spec.loader is not None
    second = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(second)

    monkeypatch.setattr(litellm, "callbacks", [second.require_guardrail])
    guardrail = _block()  # the copy's guard counts for this module's guardrail
    monkeypatch.setattr(litellm, "callbacks", [second.require_guardrail, guardrail])
    try:
        assert await _guard_lets_through(second.require_guardrail) is True
    finally:
        await _close(guardrail)


async def test_the_guards_error_line_shows_where_no_guardrail_set_the_logger_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every entry skipped, no guardrail of this worker ran `_make_visible`: the guard sets the logger up itself, so
    its line goes out with a time and a level as the others do."""
    monkeypatch.setattr(litellm, "callbacks", [])
    log = logging.getLogger("collieai_guardrail")
    log.setLevel(logging.NOTSET)  # as a worker starts; the autouse fixture gives the logger back

    assert await _guard_lets_through(RequireGuardrail()) is False

    assert log.level == logging.INFO
