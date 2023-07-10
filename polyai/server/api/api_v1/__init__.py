from flask import Flask

def run(host, port, debug):
    app = Flask(__name__, static_url_path = "")

    from . import blocking
    app.register_blueprint(blocking.bp, url_prefix="/api/v1")

    app.run(host=host, port=port, debug=debug)
