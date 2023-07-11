import asyncio
import json
from threading import Thread
from websockets.server import serve

import pylogg
import polyai.server.state as state

# Set log prefix
log = pylogg.New("api/v2/ws")

PATH = '/api/v1/stream'

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


@with_api_lock
async def _handle_stream_message(websocket, message):
    log.trace("Stream requested: {}", message)

    body = json.loads(message)
    prompt = body['prompt']
    body['stream'] = True

    # As we stream, only send the new bytes.
    skip_index = 0
    message_num = 0

    for a in state.LLM.stream(prompt, body):
        to_send = a[skip_index:]
        if to_send is None or chr(0xfffd) in to_send:  # partial unicode character, don't send it yet.
            continue

        await websocket.send(json.dumps({
            'event': 'text_stream',
            'message_num': message_num,
            'text': to_send
        }))

        await asyncio.sleep(0)
        skip_index += len(to_send)
        message_num += 1

    await websocket.send(json.dumps({
        'event': 'stream_end',
        'message_num': message_num
    }))


@with_api_lock
async def _handle_chat_stream_message(websocket, message):
    body = json.loads(message)

    user_input = body['user_input']
    body['stream'] = True

    # regenerate = body.get('regenerate', False)
    # _continue = body.get('_continue', False)
    # generator = generate_chat_reply(
    #     user_input, generate_params, regenerate=regenerate, _continue=_continue, loading_message=False)
    log.warn("Chat stream requested. Not fully supported.")

    message_num = 0
    for a in state.LLM.stream(user_input, body):
        await websocket.send(json.dumps({
            'event': 'text_stream',
            'message_num': message_num,
            'history': a
        }))

        await asyncio.sleep(0)
        message_num += 1

    await websocket.send(json.dumps({
        'event': 'stream_end',
        'message_num': message_num
    }))


async def _handle_connection(websocket, path):
    if path == PATH:
        async for message in websocket:
            await _handle_stream_message(websocket, message)

    elif path == '/api/v1/chat-stream':
        async for message in websocket:
            await _handle_chat_stream_message(websocket, message)

    else:
        log.warn(f'Unknown path requested: {path}')
        return


async def _run(host: str, port: int):
    async with serve(_handle_connection, host, port, ping_interval=None):
        await asyncio.Future()  # run forever


def _run_server(port: int, listen: bool = False):
    address = '0.0.0.0' if listen else '127.0.0.1'
    log.info(f'Starting streaming server at ws://{address}:{port}{PATH}')
    asyncio.run(_run(host=address, port=port))


def start_server(port: int, listen: bool = False):
    Thread(target=_run_server, args=[port, listen], daemon=False).start()
