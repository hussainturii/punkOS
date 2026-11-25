"""Cancellable local inference HTTP requests, including time before first byte."""
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
import httpx


class InferenceCancelled(Exception):
    pass


def _run_sync(factory):
    """Keep our HTTP loop separate from Playwright's synchronous API loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(factory())
    # Only HTTP runs on this thread; Playwright objects stay on their owner
    # thread. Join before returning so cancellation leaves no orphan request.
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix='punkos-http') as pool:
        return pool.submit(lambda: asyncio.run(factory())).result()


async def _post(url, payload, timeout_seconds, cancelled, method='POST'):
    if cancelled():
        raise InferenceCancelled('Inference cancelled')
    deadline = time.monotonic() + timeout_seconds
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_seconds, connect=10)) as client:
        request = asyncio.create_task(client.request(method, url, **({'json': payload} if method == 'POST' else {})))
        try:
            while True:
                if cancelled():
                    raise InferenceCancelled('Inference cancelled')
                if time.monotonic() >= deadline:
                    raise TimeoutError('Local inference time limit exceeded')
                done, _ = await asyncio.wait({request}, timeout=0.1)
                if done:
                    if cancelled():
                        raise InferenceCancelled('Inference cancelled')
                    response = request.result()
                    response.raise_for_status()
                    return response.json()
        finally:
            if not request.done():
                request.cancel()
            await asyncio.gather(request, return_exceptions=True)


def post_json(url, payload, timeout_seconds, cancelled):
    return _run_sync(lambda: _post(url, payload, timeout_seconds, cancelled))


def get_json(url, timeout_seconds, cancelled):
    return _run_sync(lambda: _post(url, None, timeout_seconds, cancelled, method='GET'))


async def _cloud_response(url, payload, headers, timeout, cancelled):
    """Return the HTTP response intact so providers keep their error handling."""
    if cancelled():
        raise InferenceCancelled('Inference cancelled')
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
        request = asyncio.create_task(client.post(url, json=payload, headers=headers))
        try:
            while True:
                if cancelled():
                    raise InferenceCancelled('Inference cancelled')
                done, _ = await asyncio.wait({request}, timeout=0.1)
                if done:
                    if cancelled():
                        raise InferenceCancelled('Inference cancelled')
                    return request.result()
        finally:
            if not request.done():
                request.cancel()
            await asyncio.gather(request, return_exceptions=True)


def post_response(url, payload, headers, timeout, cancelled):
    return _run_sync(lambda: _cloud_response(url, payload, headers, timeout, cancelled))
