from flask import Flask, jsonify, abort, request, make_response, url_for

def run(host, port, debug):
    app = Flask(__name__, static_url_path = "")

    from . import api
    app.register_blueprint(api.bp, url_prefix="/api")

    app.run(host=host, port=port, debug=debug)
