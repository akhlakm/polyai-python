import os
import time
import json
from threading import Thread

from flask import (
    Blueprint, jsonify, make_response, request,
    abort, session, redirect,
)

import pylogg
import polyai.server
from polyai.server import models
from polyai.server import database
from polyai.server import orm

# API version
__version__ = "1.0"

POLYAI_REQUEST_LENGTH = int(os.getenv("POLYAI_REQUEST_LENGTH") or 2048)
POLYAI_INSTRUCTION_FMT = os.getenv("POLYAI_INSTRUCTION_FMT", "")
POLYAI_USER_FMT = os.getenv("POLYAI_USER_FMT", "USER:")
POLYAI_BOT_FMT = os.getenv("POLYAI_BOT_FMT", "ASSISTANT:")

# Handle all the urls that starts with /api
bp = Blueprint("apiv1", __name__)

log = pylogg.New("endpoint")


@bp.route('/chat/completions', methods = ['POST'])
def chat_completions():
    """
    Handle the chat completion requests. Must be a post method.
    
    """
    apiKey = request.headers.get("Api-Key", None)
    len = int(request.headers.get('Content-Length') or 0)
    if len > int(POLYAI_REQUEST_LENGTH):
        abort(400, "max request length exceeded")

    if request.is_json:
        inputs = request.get_json()
        output = response_for_json(inputs)
    else:
        # if not json formatted, create a simple prompt.
        log.warning("Not a JSON request.")
        inputs = request.get_data(as_text=True)
        if not inputs:
            abort(400, "no input received")
        else:
            output = response_for_text(inputs)

    # model text generation stats
    texts, p_tok, c_tok, dt = output

    # id of the chat request
    idStr = create_idStr("chcmpl")

    # convert to openai like json format
    ch = [make_choice_dict(r, 'stop') for r in texts]
    payload = make_response_dict(idStr, 'chat.completions',
                                 polyai.server.modelName, dt,
                                 prompt_tok=p_tok, compl_tok=c_tok, choices=ch)

    # http response
    resp = make_response(jsonify(payload))

    # Add the request info to database in the background
    Thread(target=store, args=(inputs, payload, resp.headers,
                               apiKey, request.url, request.method,
                               request.headers)).start()
    
    # Respond
    return resp


@bp.route('/', methods=["GET"])
def index():
    """ Handle the get requests with a simple page. """
    message  = f"Welcome to PolyAI API v{__version__}.\n"
    message += f"Current model: {polyai.server.modelName}\n"
    message += "Please send a post request to talk to the LLM.\n"
    message += 'Shell example :\n\n\tcurl http://localhost:8080/api/chat/completions -d "hello"\n'

    return message


def response_for_json(js : dict):
    """
    Construct a instruct prompt with the request json.

    Returns:
        A tuple with the model's generated text,
        number of prompt tokens, and the number of completion tokens.
        The generated text is stripped off the BOS, EOS tokens
        and the request text.
    """
    log.trace("JSON request: {}", js)

    # Parse the json request
    messages = js.get("messages")
    prompt = js.get("prompt")
    temperature = js.get("temperature")
    max_tokens = js.get("max_tokens")
    min_tokens = js.get("min_tokens")
    top_p = js.get("top_p")

    # Create gptq parameter set
    inputs = {}
    try:
        if temperature:
            inputs['temp'] = float(temperature)
        if max_tokens:
            inputs['maxlen'] = int(max_tokens)
        if min_tokens:
            inputs['minlen'] = int(min_tokens)
        if top_p:
            inputs['top_p'] = float(top_p)
    except:
        abort(400, "invalid model parameter type")

    # If a promt is given, ignore the messages
    if prompt:
        if 'assistant:' in prompt.lower():
            # already formatted
            message = prompt
        else:
            # format using env vars
            message = f"{POLYAI_USER_FMT} {prompt} {POLYAI_BOT_FMT}"

    else:
        # parse the messages
        # messasges = [ {'role': '', content: ''}]
        if messages is None:
            abort(400, "list of messages not provided")

        if not type(messages) == list:
            abort(400, "messages must be a list")

        # build the model prompt from the messages
        message = ""
        instruction = ""
        for m in messages:
            if not type(m) == dict:
                abort(400, "message element must be of format {'role': "", 'content': ""}")

            role = m.get('role')
            cont = m.get('content')
            if role is None or cont is None:
                abort(400, "message element must be of format {'role': "", 'content': ""}")

            role = str(role).lower()
            cont = str(cont)

            if role == "system":
                # A system role is the main instruction
                instruction += f"{POLYAI_INSTRUCTION_FMT} {cont}\n".lstrip()
            else:
                # Few-shots
                if role == "user": role = POLYAI_USER_FMT
                elif role == "assistant": role = POLYAI_BOT_FMT
                message += f"{role} {cont}\n".lstrip()

        # add the final string for assistant
        message = f"{instruction}{message}{POLYAI_BOT_FMT}"


    inputs['prompt'] = message
    responses, p_tok, c_tok, dt = models.get_gptq_response(**inputs)

    # remove the start end <s> tokens, and strip the input prompt
    # @todo: this depends on the model
    for i in range(len(responses)):
        #responses[i] = responses[i][4:-4].replace(message, "", 1).strip()
        responses[i] = responses[i][3:].replace(message, "", 1).strip()

    return responses, p_tok, c_tok, dt


def response_for_text(text : str):
    """
    Construct a simple instruct prompt with the request text.

    Returns:
        A tuple with the model's generated text,
        number of prompt tokens, and the number of completion tokens.
        The generated text is stripped off the BOS, EOS tokens
        and the request text.
    """
    log.trace("Text request: {}", text)
    message = f"{POLYAI_USER_FMT} {text} {POLYAI_BOT_FMT}"
    responses, p_tok, c_tok, dt = models.get_gptq_response(message)

    # remove the start end <s> tokens
    # and strip the input prompt
    for i in range(len(responses)):
        responses[i] = responses[i][4:-4].replace(message, "", 1).strip()

    return responses, p_tok, c_tok, dt


def make_response_dict(idstr : str, object : str, model : str, dt : int,
               prompt_tok : int, compl_tok : int, choices : list):
    
    # set choice indices
    for i, ch in enumerate(choices):
        choices[i]['index'] = 0

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
        'choices': choices
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
        Ignore with a warning if db connection fails.
    """
    try:
        db = database.connect()
    except Exception as err:
        log.warning("Failed to connect database: {}", err)
        return

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


def create_idStr(prefix):
    """ Use milliseconds to create a unique id. """
    idStr = str(round(time.time() * 1000))
    return prefix + "-" + idStr


@bp.errorhandler(400)
def bad_request(error):
    errstr = str(error)
    log.info(errstr)
    return make_response(errstr, 400)


@bp.errorhandler(404)
def not_found(error):
    errstr = str(error)
    log.info(errstr)
    return make_response(errstr, 404)
