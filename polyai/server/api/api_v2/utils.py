import asyncio
import functools
import threading

# We use a thread local to store the asyncio lock, so that each thread
# has its own lock.  This isn't strictly necessary, but it makes it
# such that if we can support multiple worker threads in the future,
# thus handling multiple requests in parallel.
api_tls = threading.local()


def _get_api_lock(tls) -> asyncio.Lock:
    """
    The streaming and blocking API implementations each run on their own
    thread, and multiplex requests using asyncio. If multiple outstanding
    requests are received at once, we will try to acquire the shared lock
    shared.generation_lock multiple times in succession in the same thread,
    which will cause a deadlock.

    To avoid this, we use this wrapper function to block on an asyncio
    lock, and then try and grab the shared lock only while holding
    the asyncio lock.
    """
    if not hasattr(tls, "asyncio_lock"):
        tls.asyncio_lock = asyncio.Lock()

    return tls.asyncio_lock


def with_api_lock(func):
    """
    This decorator should be added to all streaming API methods which
    require access to the shared.generation_lock.  It ensures that the
    tls.asyncio_lock is acquired before the method is called, and
    released afterwards.
    """
    @functools.wraps(func)
    async def api_wrapper(*args, **kwargs):
        async with _get_api_lock(api_tls):
            return await func(*args, **kwargs)
    return api_wrapper
