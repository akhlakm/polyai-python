import os
import sys
import dotenv
import argparse
import pylogg as log

import polyai
from polyai.server import models
from polyai.server import app


model = None
token = None
pipeline = None
modelName = None

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

    if not dotenv.load_dotenv():
        raise RuntimeError("ENV not loaded")

    log.setFile(open("polyai.log", "a+"))
    log.setConsoleTimes(show=True)

    if args.cmd == "server":
        if args.model is None:
            modelpath = os.path.join("models", os.environ.get("POLYAI_MODEL_PATH"))
            if os.path.exists(modelpath):
                args.model = modelpath

        if args.model is not None:
            try:
                models.init_gptq_model(args.model)
            except:
                models.init_hf_bert(args.model)
        else:
            log.error("No valid modelpath specified. Models can specified using the --model argument.")
            log.error("Alternatively, set the POLYAI_MODEL_PATH relative to ./models/ directory.")

        log.info("Running server on {}:{}", args.host, args.port)
        app.run(args.host, args.port, debug=False)

    log.close()
    return 0

if __name__ == "__main__":
    sys.exit(main())
