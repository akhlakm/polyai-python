from flask import Flask, jsonify, abort, request, make_response, url_for

def run(host, port, debug):
    app = Flask(__name__, static_url_path = "")

    from . import api_v1 as api
    app.register_blueprint(api.bp, url_prefix="/api/v1")

    app.run(host=host, port=port, debug=debug)
