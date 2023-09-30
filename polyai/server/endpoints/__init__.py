from flask import Flask
from polyai.server.endpoints import openai
from polyai.server.endpoints import textgen

def run(polyai_port, streaming_port, listen=False, ssl=False, debug=False):
    host = '0.0.0.0' if listen else '127.0.0.1'
    app = Flask(__name__, static_url_path = "")

    print(f"Serving openai API at /polyai")
    print(f"Serving textgen API at /api")

    app.register_blueprint(openai.blocking.bp, url_prefix="/polyai/")
    app.register_blueprint(textgen.blocking.bp, url_prefix="/api/v1/")
    textgen.streaming.start_server(streaming_port, listen)

    if ssl:
        app.run(host=host, port=polyai_port, debug=False,
            ssl_context=("keys/ssl.crt", "keys/ssl.key")
        )
    else:
        app.run(host=host, port=polyai_port, debug=False)
