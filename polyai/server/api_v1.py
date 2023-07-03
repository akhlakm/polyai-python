import os
from threading import Thread

from flask import (
    Blueprint, jsonify, make_response, request,
    abort, session, redirect,
)

import pylogg
from polyai.server import models
from polyai.server import utils

# API version
__version__ = "1.0"

POLYAI_REQUEST_LENGTH = int(os.getenv("POLYAI_REQUEST_LENGTH") or 2048)
POLYAI_INSTRUCTION_FMT = os.getenv("POLYAI_INSTRUCTION_FMT", "")
POLYAI_USER_FMT = os.getenv("POLYAI_USER_FMT", "USER:")
POLYAI_BOT_FMT = os.getenv("POLYAI_BOT_FMT", "ASSISTANT:")

# Handle all the urls that starts with /api
bp = Blueprint("apiv1", __name__)

log = pylogg.New("endpoint")


@bp.route('/', methods=["GET"])
def index():
    """ Handle the get requests with a simple page. """
    message  = f"Welcome to PolyAI API v{__version__}.\n"
    message += "Please send a post request to talk to the LLM.\n"
    message += 'Shell example :\n\n\tcurl http://localhost:8080/api/chat/completions -d "hello"\n'

    return message


@bp.route('/bert/ner', methods = ['POST'])
def bert_ner():
    """
    Handle NER requests. Must be a post method.
    
    """
    apiKey = request.headers.get("Api-Key", None)

    if request.is_json:
        js = request.get_json()
        text = js.get("text")
    else:
        text = request.get_data(as_text=True)
        if not text:
            abort(400, "no input received")

    mname, ner, dt = models.get_bert_ner(text)
    p_tok = 0
    c_tok = 0

    # id of the chat request
    idStr = utils.create_idStr("ner")

    # convert to openai like json format
    payload = utils.make_response_dict(idStr, 'bert.ner', mname, dt,
                                       prompt_tok=p_tok, compl_tok=c_tok, ner=ner)

    # http response
    resp = make_response(jsonify(payload))

    # Add the request info to database in the background
    Thread(target=utils.store, args=(text, payload, resp.headers,
                               apiKey, request.url, request.method,
                               request.headers)).start()
    
    # Respond
    return resp



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
    model_name, texts, p_tok, c_tok, dt = output

    assert type(texts) == list, "model response must be a list of str"

    # id of the chat request
    idStr = utils.create_idStr("chcmpl")

    # convert to openai like json format
    ch = [utils.make_choice_dict(r, 'stop') for r in texts]
    payload = utils.make_response_dict(idStr, 'chat.completions', model_name, dt,
                                       prompt_tok=p_tok, compl_tok=c_tok, choices=ch)

    # http response
    resp = make_response(jsonify(payload))

    # Add the request info to database in the background
    Thread(target=utils.store, args=(inputs, payload, resp.headers,
                               apiKey, request.url, request.method,
                               request.headers)).start()
    
    # Respond
    return resp


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

    # Check types
    validate('temperature', float, js)
    validate('max_tokens', int, js)
    validate('min_tokens', int, js)
    validate('prompt', str, js)
    validate('top_p', float, js)
    validate('top_k', int, js)

    # Parse the json request
    messages = js.get("messages")
    prompt = js.get("prompt")
    
    # If a prompt is given, ignore the messages
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

    js['prompt'] = message

    try:
        response = models.get_exllama_response(stream=False, **js)
    except ConnectionError:
        abort(409, "Model not ready.")
    return response


def validate(name : str, ptype : callable, d : dict):
    """ Validate a dictionary item by typecasting. """
    value = d.get(name, None)
    if value is not None:
        try:
            value = ptype(value)
        except:
            abort(400, f"invalid type for {name}")
    d[name] = value
    return d


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
    try:
        response = models.get_exllama_response(message, stream=False)
    except ConnectionError:
        abort(409, "Model not ready.")
    return response


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
