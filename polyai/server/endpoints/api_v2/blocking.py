import os
import ssl
import json
from flask import (
    Blueprint, jsonify, make_response, request,
    abort, session, redirect,
)

from threading import Thread

import pylogg
import polyai.server.state as state

# Set log prefix
log = pylogg.New("api")

# Handle all the urls that starts with /api/v1
bp = Blueprint("api", __name__)

def get_model_info():
    return {
        'model_name': state.LLM.model_name(),
        'lora_names': [state.LLM._lora_name],
        # dump
        'shared.settings': {},
        'shared.args': [],
    }

def respond(obj : dict = {}, status = 200):
    resp = make_response(jsonify(obj), status)
    resp.headers['Access-Control-Allow-Origin'] =  '*'
    resp.headers['Access-Control-Allow-Methods'] =  '*'
    resp.headers['Access-Control-Allow-Headers'] =  '*'
    resp.headers['Cache-Control'] =  'no-store, no-cache, must-revalidate'
    return resp


@bp.route('/', methods=["GET"])
def index():
    return respond({
        'result': 'Server OK.'
    })


@bp.route('/', methods=["OPTIONS"])
def options():
    return respond({})


@bp.route('/generate', methods=["POST"])
def generate():
    if not request.is_json:
        abort(400)
    body = request.get_json()
    prompt = body['prompt']
    try:
        output = state.LLM.generate(prompt, body)
        model, reply_list, ptok, ctok, dt = output
    except Exception as err:
        reply_list = [str(err)]
    return respond({
        'results': [{
            'text': "\n".join(reply_list)
        }]
    })


@bp.route('/chat', methods=["POST"])
def chat():
    if not request.is_json:
        abort(400)
    body = request.get_json()
    user_input = body['user_input']
    body['stream'] = False

    # Chat reply
    # regenerate = body.get('regenerate', False)
    # _continue = body.get('_continue', False)
    # generator = generate_chat_reply(
    #     user_input, generate_params, regenerate=regenerate, _continue=_continue, loading_message=False)
    # answer = generate_params['history']
    log.warn("Chat generation requested. Not fully supported.")

    output = state.LLM.generate(prompt, body)
    model, reply_list, ptok, ctok, dt = output

    return respond({
        'results': [{
            'history': "\n".join(reply_list)
        }]
    })


@bp.route('/stop-stream', methods=["POST"])
def stop_stream():
    log.trace("Stop generation requested.")
    state.LLM.stop_generation()
    return respond({
        'results': 'success'
    })


@bp.route('/token-count', methods=["POST"])
def token_count():
    if not request.is_json:
        abort(400)
    body = request.get_json()
    log.trace("Token count requested: {}", body['prompt'])
    tokens = state.LLM.encode(body['prompt'])[0]
    return respond({
        'results': [{
            'tokens': len(tokens)
        }]
    })


@bp.route('/model', methods=["GET"])
def model():
    return respond({
        'result': state.LLM.model_name() or "No model loaded"
    })

@bp.route('/model', methods=["POST"])
def action():
    if not request.is_json:
        abort(400)
    body = request.get_json()

    # by default return the same as the GET interface
    result = state.LLM.model_name()

    # Actions: info, load, list, unload
    action = body.get('action', '')

    if action == 'load':
        model_name = body['model_name']
        args = body.get('args', {})
        print('args', args)

        response = json.dumps({'error': {'message': 'not allowed'}})
        self.wfile.write(response.encode('utf-8'))
        abort(403, "Permission denied - model load")

    elif action == 'unload':
        response = json.dumps({'error': {'message': 'not allowed'}})
        self.wfile.write(response.encode('utf-8'))
        abort(403, "Permission denied - model unload")

    elif action == 'list':
        log.trace("Model list requested.")
        result = [state.LLM.model_name()]

    elif action == 'info':
        log.trace("Model info requested.")
        result = {
            'model_name': state.LLM.model_name(),
            'lora_names': [state.LLM._lora_name],
        }

    return respond({
        'result': result,
    })


@bp.errorhandler(400)
def bad_request(error):
    errstr = str(error)
    log.info(errstr)
    return respond({
        'result': errstr
    }, 400)


@bp.errorhandler(404)
def not_found(error):
    errstr = str(error)
    log.info(errstr)
    return respond({
        'result': errstr
    }, 404)
