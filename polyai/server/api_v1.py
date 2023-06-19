import time

from flask import (
    Blueprint, jsonify, make_response, request,
    abort, session, redirect,
)

import polyai.server
from polyai.server import models

__version__ = "1.0"

MAX_CONTEXT_LENGTH = 2048

bp = Blueprint("apiv1", __name__)


@bp.route('/chat/completions', methods = ['POST'])
def chat_completions():
    len = int(request.headers.get('Content-Length') or 0)
    if len > MAX_CONTEXT_LENGTH:
        abort(400, message="max context length exceeded")

    message = request.get_data(as_text=True)
    if not message:
        abort(400)
    else:
        response, p_tok, c_tok = models.get_gptq_response(message)
        ch = make_choice_dict(response, 'stop')
        return jsonify(make_response_dict("test", 'chat.completions', 'test_model',
                                  prompt_tok=p_tok, compl_tok=c_tok, choices=[ch]))


@bp.route('/', methods=["GET"])
def index():
    message = f"Welcome to PolyAI API version 1.0. Current model: {polyai.server.model}"
    message += "Please send a post request to talk to the LLM.\n"
    message += 'Shell example :\n\n\tcurl http://localhost:8080/api/chat/completions -d "hello"\n'
    message += "\nUse the 'USER: <prompt> ASSISTANT: ' format.\n\n"

    return message


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
    return make_response(jsonify( { 'error': 'Bad request', 'description': error.message } ), 400)


@bp.errorhandler(404)
def not_found(error):
    return make_response(jsonify( { 'error': 'Not found' } ), 404)

