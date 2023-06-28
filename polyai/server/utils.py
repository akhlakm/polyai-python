import time
import json

from polyai.server import database
from polyai.server import orm


def make_response_dict(idstr : str, object : str, model : str, dt : int,
               prompt_tok : int, compl_tok : int, choices : list = [],
               ner : list = []):
    
    """ Construct and return the response dict for json payload. """
    
    if len(ner) == 0 and len(choices) == 0:
        raise ValueError("Either ner or choices need to be provided")
    
    # set indices
    for i, ch in enumerate(choices):
        choices[i]['index'] = i

    for i, ch in enumerate(ner):
        ner[i]['index'] = i

    return {
        'id': idstr,
        'object': object,
        'created': time.strftime("%a, %b %d %Y %X"),
        'model': model,
        'usage': {
            'prompt_tokens': prompt_tok,
            'completion_tokens': compl_tok,
            'total_tokens': prompt_tok + compl_tok
        },
        'elapsed_msec': dt,
        'choices': choices,
        'ner_tags': ner,
    }


def make_choice_dict(response, finish_reason):
    # openai like response object
    return {
        'message': {
            'role': 'assistant',
            'content': response,
        },
        'finish_reason': finish_reason
    }


def store(message, respObj, respheads, apiKey, url, method, reqheads):
    """ Store api request and response info to database.
    Returns:
        True if successful.
    """
    try:
        db = database.connect()
    except Exception as err:
        return False

    output = " || ".join([ch['message']['content']
                            for ch in respObj['choices']])
    if type(message) == dict:
        message = json.dumps(message)

    apiReq = orm.APIRequest(
        idStr = respObj['id'],
        apikey = apiKey,
        requrl = url,
        reqmethod = method,
        model = respObj['model'],
        request = message,
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


def create_idStr(prefix):
    """ Use milliseconds to create a unique id. """
    idStr = str(round(time.time() * 1000))
    return prefix + "-" + idStr

