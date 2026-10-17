"""One POST under one absolute deadline.

No retry anywhere: the client's transport is built with `retries=0`, redirects are not followed, and nothing here
loops. The deadline is a point on the event loop's clock fixed when the phase starts; the whole exchange — a
connection from the pool, the connect, the send and the reply read in full — runs under `asyncio.timeout_at` of it.
httpx's own timeouts are set to the time left as a second line, not as the bound.

The reply is read raw, with `Accept-Encoding: identity`, and at most `max_reply_bytes` of it.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass

import httpx

#: The pool: the server evaluates one call at a time per worker and answers a busy worker at
#: once, so a deep pool only queues work the deadline would cancel.
POOL_LIMITS = httpx.Limits(max_connections=64, max_keepalive_connections=16)


@dataclass(frozen=True)
class Reply:
    status: int
    body: bytes


@dataclass(frozen=True)
class Failure:
    """No reply: `reason` is `timeout`, `transport` or `protocol` (a reply over the size limit); `detail` names the
    exception type or the condition, never a message."""

    reason: str
    detail: str


def new_client(*, transport: httpx.AsyncBaseTransport | None = None) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=transport if transport is not None else httpx.AsyncHTTPTransport(retries=0, limits=POOL_LIMITS),
        follow_redirects=False,
    )


async def post_once(
    client: httpx.AsyncClient,
    *,
    url: str,
    headers: dict[str, str],
    body: bytes,
    deadline: float,
    max_reply_bytes: int,
    on_attempt: Callable[[], None] | None = None,
) -> Reply | Failure:
    """`on_attempt` is called when the transport attempt starts, before the connection — a refused connection is an
    attempt — and never when no time was left to start it. It does not say that any byte reached the server."""
    loop = asyncio.get_running_loop()
    remaining = deadline - loop.time()
    if remaining <= 0:
        return Failure("timeout", "no_time_left")
    if on_attempt is not None:
        on_attempt()
    try:
        async with asyncio.timeout_at(deadline):
            async with client.stream(
                "POST",
                url,
                content=body,
                headers={**headers, "Accept-Encoding": "identity"},
                timeout=httpx.Timeout(remaining),
            ) as response:
                chunks: list[bytes] = []
                size = 0
                async for chunk in response.aiter_raw():
                    size += len(chunk)
                    if size > max_reply_bytes:
                        return Failure("protocol", "reply_too_large")
                    chunks.append(chunk)
                return Reply(response.status_code, b"".join(chunks))
    except TimeoutError:
        return Failure("timeout", "deadline")
    except httpx.TimeoutException as error:
        return Failure("timeout", type(error).__name__)
    except httpx.HTTPError as error:
        return Failure("transport", type(error).__name__)
