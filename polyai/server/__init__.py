import os
import sys
import argparse
import pylogg as log

import polyai
from polyai.server import models
from polyai.server import app


model = None
token = None

def parse_arguments():
    parser = argparse.ArgumentParser(description="PolyAI Server (v%s)" %polyai.__version__)

    parser.add_argument("cmd", help="server")

    parser.add_argument("--host", default="0.0.0.0", help="server host IP address")
    parser.add_argument("--port", default=8080, type=int, help="server port")
    parser.add_argument("--model", default=None, help="LLM model to load")

    parser.add_argument("--debug", default=True, action='store_true',
                        help="enable debugging")

    args = parser.parse_args()

    return args


def main():
    args = parse_arguments()
    if args.debug:
        log.setLevel(log.DEBUG)

    log.setFile(open("polyai.log", "a+"))
    log.setConsoleTimes(show=True)

    if args.cmd == "server":
        if args.model is not None:
            models.MODELDIR = os.environ.get("POLYAI_MODEL_DIR", None)
            if models.MODELDIR is None:
                raise RuntimeError("Cannot detect models directory.")
            models.init_gptq_model(args.model)

        log.info("Running server on host={}:{}", args.host, args.port)
        app.run(args.host, args.port, args.debug)

    log.close()
    return 0

if __name__ == "__main__":
    sys.exit(main())
