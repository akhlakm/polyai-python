from flask import Flask
from polyai.server.api import api_v1
from polyai.server.api import api_v2

def run(polyai_port, streaming_port, listen=False, ssl=False, debug=False):
    host = '0.0.0.0' if listen else '127.0.0.1'
    app = Flask(__name__, static_url_path = "")

    print(f"Serving openai API at /polyai")
    print(f"Serving textgen API at /api")

    app.register_blueprint(api_v1.blocking.bp, url_prefix="/polyai/")
    app.register_blueprint(api_v2.blocking.bp, url_prefix="/api/v1/")
    api_v2.streaming.start_server(streaming_port, listen)

    if ssl:
        app.run(host=host, port=polyai_port, debug=False,
            ssl_context=("keys/ssl.crt", "keys/ssl.key")
        )
    else:
        app.run(host=host, port=polyai_port, debug=False)
