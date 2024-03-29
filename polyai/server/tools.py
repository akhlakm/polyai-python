import time
import json
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

