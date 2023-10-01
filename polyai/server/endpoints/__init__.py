import pylogg
from flask import Flask
from polyai.server.endpoints import openai
from polyai.server.endpoints import textgen

def run(polyai_port, streaming_port, listen=False, ssl=False, debug=False):
    log = pylogg.New('endpoint')
    protocol = 'https' if ssl else 'http'
    host = '0.0.0.0' if listen else '127.0.0.1'
    app = Flask(__name__, static_url_path = "")

    app.register_blueprint(openai.blocking.bp, url_prefix="/polyai/")
    log.note('OpenAI like API endpoint {}://{}:{}{}',
             protocol, host, polyai_port, "/polyai")

    app.register_blueprint(textgen.blocking.bp, url_prefix="/api/v1/")
    log.note('TextGen like API endpoint {}://{}:{}{}',
             protocol, host, polyai_port, "/api/v1")

    # This runs in the bg, so start first.
    textgen.streaming.start_server(streaming_port, listen)

    # Start the main blocking server.
    if ssl:
        app.run(host=host, port=polyai_port, debug=False,
            ssl_context=("keys/ssl.crt", "keys/ssl.key")
        )
    else:
        app.run(host=host, port=polyai_port, debug=False)
