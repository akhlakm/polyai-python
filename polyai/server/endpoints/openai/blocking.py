from flask import (
    Blueprint, jsonify, make_response, request,
    abort, session, redirect,
)

import pylogg
import polyai.sett as sett
import polyai.server.state as state

from polyai.server import tools
from polyai.server.endpoints.openai import utils

# Handle all the urls that starts with /polyai
bp = Blueprint("polyai", __name__)

# Set log prefix
log = pylogg.New("end1")

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

    mname, ner, dt = state.BERT.ner_tags(text)
    p_tok = 0
    c_tok = 0

    # id of the chat request
    idStr = tools.create_idStr("ner")

    # convert to openai like json format
    payload = utils.make_response_dict(idStr, 'bert.ner', mname, dt,
                                       prompt_tok=p_tok, compl_tok=c_tok, ner=ner)

    # http response
    resp = make_response(jsonify(payload))

    # Add the request info to database in the background
    tools.store(text, payload, resp.headers, apiKey, request.url,
                      request.method, request.headers)

    # Respond
    return resp


@bp.route('/text/embedding', methods = ['POST'])
def text_embeddings():
    """
    Handle Embedding requests. Must be a post method.
    
    """
    apiKey = request.headers.get("Api-Key", None)

    if request.is_json:
        js = request.get_json()
        text = js.get("text")
    else:
        text = request.get_data(as_text=True)
        if not text:
            abort(400, "no input received")

    t2 = log.trace("Calculating text embeddings.")

    inputs = state.LLM.encode(text)
    inputs = inputs[-1].tolist()

    p_tok = len(inputs)
    dt = t2.elapsed()

    c_tok = 0
    model = state.LLM.model_name()

    # id of the chat request
    idStr = tools.create_idStr("embedding")

    # convert to openai like json format
    payload = utils.make_response_dict(idStr, 'text.embedding', model, dt,
                                       prompt_tok=p_tok, compl_tok=c_tok,
                                       embeddings=inputs)

    # http response
    resp = make_response(jsonify(payload))

    # Add the request info to database in the background
    tools.store(text, payload, resp.headers, apiKey, request.url,
                      request.method, request.headers)

    # Respond
    return resp


@bp.route('/chat/completions', methods = ['POST'])
def chat_completions():
    """
    Handle the chat completion requests. Must be a post method.
    
    """
    apiKey = request.headers.get("Api-Key", None)
    len = int(request.headers.get('Content-Length') or 0)
    if len > sett.Server.max_content_len:
        abort(400, "max content length exceeded")

    if request.is_json:
        inputs = request.get_json()
        output = response_for_json(inputs)
    else:
        abort(400, "request must be valid JSON formatted")

    # model text generation stats
    model_name, texts, p_tok, c_tok, dt = output

    assert type(texts) == list, "model response must be a list of str"

    # id of the chat request
    idStr = tools.create_idStr("chcmpl")

    # convert to openai like json format
    ch = [utils.make_choice_dict(r, 'stop') for r in texts]
    payload = utils.make_response_dict(idStr, 'chat.completions', model_name, dt,
                                       prompt_tok=p_tok, compl_tok=c_tok, choices=ch)

    # http response
    resp = make_response(jsonify(payload))

    # Add the request info to database in the background
    tools.store(inputs, payload, resp.headers, apiKey,
                    request.url, request.method, request.headers)

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
    validate('max_length', int, js)
    validate('min_tokens', int, js)
    validate('prompt', str, js)
    validate('top_p', float, js)
    validate('top_k', int, js)

    # Parse the json request
    messages = js.get("messages")
    prompt = js.get("prompt")

    max_tokens = js.get('max_length')
    if max_tokens is None:
        max_tokens = js.get('max_tokens') or sett.TextGen.context_length

    # If a prompt is given, ignore the messages
    if prompt:
        if 'assistant:' in prompt.lower():
            # already formatted
            message = prompt
        else:
            # format using env vars
            message = f"{sett.TextGen.user_fmt} {prompt} {sett.TextGen.bot_fmt}"

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
                instr = sett.TextGen.instruction_fmt
                if "{content}" in instr:
                    instruction = instr.format(content=cont).lstrip()+"\n"
                else:
                    instruction += f"{instr} {cont}\n".lstrip()
            else:
                # Few-shots
                if role == "user": role = sett.TextGen.user_fmt
                elif role == "assistant": role = sett.TextGen.bot_fmt
                message += f"{role} {cont}\n".lstrip()

        # add the final string for assistant
        message = f"{instruction}{message}{sett.TextGen.bot_fmt}"

    # Check if message is not too big.
    inputs = state.LLM.encode(message)[-1]

    if len(inputs) > sett.TextGen.context_length:
        abort(400, "max request tokens length exceeded")

    if len(inputs) + max_tokens >= sett.TextGen.context_length:
        js['max_tokens'] = sett.TextGen.context_length - len(inputs) - 1
        js['max_new_tokens'] = js['max_tokens']
        log.warn("Set allowed max_tokens from {} to {}",
                 max_tokens, js.get('max_tokens'))

    try:
        response = state.LLM.generate(message, js)
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
