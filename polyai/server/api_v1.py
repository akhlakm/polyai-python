import time

from flask import (
    Blueprint, jsonify, make_response, request,
    abort, session, redirect,
)

import pylogg
import polyai.server
from polyai.server import models

__version__ = "1.0"
MAX_CONTEXT_LENGTH = 1024

bp = Blueprint("apiv1", __name__)
log = pylogg.New("endpoint")

@bp.route('/chat/completions', methods = ['POST'])
def chat_completions():
    len = int(request.headers.get('Content-Length') or 0)
    if len > MAX_CONTEXT_LENGTH:
        abort(400, "max context length exceeded")

    if request.is_json:
        resp = response_for_json(request.get_json())
    else:
        # if not json formatted, create a simple prompt.
        message = request.get_data(as_text=True)
        if not message:
            abort(400, "no input received")
        else:
            resp = response_for_text(message)

    responses, p_tok, c_tok = resp
    ch = [make_choice_dict(r, 'stop') for r in responses]
    return jsonify(make_response_dict("test", 'chat.completions', 'test_model',
                            prompt_tok=p_tok, compl_tok=c_tok, choices=ch))


@bp.route('/', methods=["GET"])
def index():
    message = f"Welcome to PolyAI API v{__version__}. Current model: {polyai.server.model}"
    message += "Please send a post request to talk to the LLM.\n"
    message += 'Shell example :\n\n\tcurl http://localhost:8080/api/chat/completions -d "hello"\n'

    return message


def response_for_json(js : dict):
    abort(400, "not implemented")

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
    message = f"USER: {text} ASSISTANT:"
    responses, p_tok, c_tok = models.get_gptq_response(message)

    # remove the start end <s> tokens
    # and strip the input prompt
    for i in range(len(responses)):
        responses[i] = responses[i][4:-4].replace(message, count=1)

    return responses, p_tok, c_tok


def make_response_dict(idstr : str, object : str, model : str,
               prompt_tok : int, compl_tok : int, choices : list):
    
    # set choice indices
    for i, ch in enumerate(choices):
        choices[i]['index'] = 0

    return {
        'id': idstr,
        'object': object,
        'created': round(time.time() * 1000),
        'model': model,
        'usage': {
            'prompt_tokens': prompt_tok,
            'completion_tokens': compl_tok,
            'total_tokens': prompt_tok + compl_tok
        },
        'choices': choices
    }


def make_choice_dict(response, finish_reason):
    return {
        'message': {
            'role': 'assistant',
            'content': response,
        },
        'finish_reason': finish_reason
    }


@bp.errorhandler(400)
def bad_request(error):
    log.info(error)
    return make_response(str(error), 400)


@bp.errorhandler(404)
def not_found(error):
    log.info(error)
    return make_response(str(error), 404)
