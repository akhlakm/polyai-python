from flask import Flask
from polyai.server.endpoints.api_v1 import blocking

def run(port, listen=False, debug=False):
    """ Run api/v1 server (openai like). """
    host = '0.0.0.0' if listen else '127.0.0.1'
    app = Flask(__name__, static_url_path = "")
    app.register_blueprint(blocking.bp, url_prefix="/api/v1/")
    app.run(host=host, port=port, debug=debug)
