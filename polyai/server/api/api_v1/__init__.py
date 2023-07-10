from flask import Flask
from polyai.server.api.api_v1 import blocking

def run(host, port, debug):
    """ Run api/v1 server (openai like). """
    app = Flask(__name__, static_url_path = "")
    app.register_blueprint(blocking.bp, url_prefix="/api/v1")
    app.run(host=host, port=port, debug=debug)
