from flask import (
    Blueprint, jsonify, make_response, request,
    abort, session, redirect,
)

bp = Blueprint("apiv1", __name__, url_prefix="/v1")


@bp.errorhandler(400)
def bad_request(error):
    return make_response(jsonify( { 'error': 'Bad request' } ), 400)

@bp.errorhandler(404)
def not_found(error):
    return make_response(jsonify( { 'error': 'Not found' } ), 404)

@bp.route('/', methods=["GET"])
def index():
    return "Welcome to API version 1.0"

@bp.route('/<int:task_id>', methods = ['POST'])
def update_task(task_id):
    task = []
    if len(task) == 0:
        abort(404)
    if not request.json:
        abort(400)
    if 'title' in request.json and type(request.json['title']) != str:
        abort(400)
    if 'description' in request.json and type(request.json['description']) is not str:
        abort(400)
    if 'done' in request.json and type(request.json['done']) is not bool:
        abort(400)
    task[0]['title'] = request.json.get('title', task[0]['title'])
    task[0]['description'] = request.json.get('description', task[0]['description'])
    task[0]['done'] = request.json.get('done', task[0]['done'])
    return jsonify( { 'task': [] } )
