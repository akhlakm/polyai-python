import time
import json
import torch

from threading import Thread

from polyai.server import database
from polyai.server import orm


def store(message, respObj, respheads, apiKey, url, method, reqheads):
    """ Store api request and response info to database. """

    def _background():
        try:
            db = database.connect()
        except Exception as err:
            return False

        output = " || ".join([ch['message']['content']
                                for ch in respObj['choices']])

        if type(message) == dict:
            reqtext = json.dumps(message)
        else:
            reqtext = message

        apiReq = orm.APIRequest(
            idStr = respObj['id'],
            apikey = apiKey,
            requrl = url,
            reqmethod = method,
            model = respObj['model'],
            request = reqtext,
            output = output,
            response = respObj,
            reqheaders = dict(reqheads),
            respheaders = dict(respheads),
            elapsed_msec = respObj['elapsed_msec'],
            request_tokens = respObj['usage']['prompt_tokens'],
            response_tokens = respObj['usage']['completion_tokens'],
        )
        apiReq.insert(db)
        return True

    Thread(target=_background).start()


def create_idStr(prefix):
    """ Use milliseconds to create a unique id. """
    idStr = str(round(time.time() * 1000))
    return prefix + "-" + idStr


def vram_usage():
    """ Return the current vram usage.
    Returns:
        current usage, total vram, free vram in GBytes.
    """
    free = 0
    total = 0
    devices = torch.cuda.device_count()
    for i in range(devices):
        mem = torch.cuda.mem_get_info(i)
        free += mem[0]/1024/1024/1024
        total += mem[1]/1024/1024/1024
    
    # print("VRAM usage, %d GPUs: %0.4f GB / %0.4f GB (%0.4f GB free)" %(devices, total-free, total, free))

    # used, total, free
    return total - free, total, free
