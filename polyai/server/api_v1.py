from flask import (
    Blueprint, jsonify, make_response, request,
    abort, session, redirect,
)

import polyai.server
from polyai.server import models

__version__ = "1.0"
bp = Blueprint("apiv1", __name__)

def choices_v1(response, finish_reason):
    return {
        'message': {
            'role': 'assistant',
            'content': response,
        },
        'finish_reason': finish_reason
    }

def reponse_v1(idstr : str, object : str, model : str,
               prompt_tok : int, compl_tok : int, choices : list):
    
    # set choice indices
    for i, ch in enumerate(choices):
        choices[i]['index'] = 0

    return {
        'id': idstr,
        'object': object,
        'created': 1677649420,
        'model': model,
        'usage': {
            'prompt_tokens': prompt_tok,
            'completion_tokens': compl_tok,
            'total_tokens': prompt_tok + compl_tok
        },
        'choices': choices
    }


@bp.errorhandler(400)
def bad_request(error):
    return make_response(jsonify( { 'error': 'Bad request' } ), 400)

@bp.errorhandler(404)
def not_found(error):
    return make_response(jsonify( { 'error': 'Not found' } ), 404)

@bp.route('/', methods=["GET"])
def index():
    message = f"Welcome to PolyAI API version 1.0. Current model: {polyai.server.model}"
    message += "Please send a post request to talk to the LLM.\n"
    message += 'Shell example :\n\n\tcurl -X POST http://localhost:8080/api/chat/completions -d "hello"\n'
    message += "\nUse the 'USER: <prompt> ASSISTANT: ' format.\n\n"

    return message

@bp.route('/chat/completions', methods = ['POST'])
def chat_completions():
    message = request.get_data(as_text=True)
    if not message:
        abort(400)
    else:
        response = models.get_gptq_response(message)
        ch = choices_v1(response, 'stop')
        return jsonify(reponse_v1("test", 'chat.completions', 'test_model', -1, -1, [ch]))
