import sys
import argparse

import pylogg as log
from polyai_server import server

model = None

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd",
                        help="server")

    parser.add_argument("--host", default="0.0.0.0", help="server host IP address")
    parser.add_argument("--port", default=8080, type=int, help="server port")
    parser.add_argument("--model", default=None, help="LLM model to load")
    parser.add_argument("--debug", default=True, action='store_true',
                        help="enable debugging")

    args = parser.parse_args()

    return args


def load_model(modelname):
    t1 = log.trace("Loading model {}", modelname)
    global model
    model = modelname #@todo
    t1.done("Model {} loaded.", modelname)


def main():
    args = parse_arguments()
    if args.debug:
        log.setLevel(log.DEBUG)

    log.setFile(open("polyai_server.log", "a+"))
    log.setConsoleTimes(show=True)

    if args.model is not None:
        load_model(args.model)

    log.info("Running server on host={}:{}", args.host, args.port)
    server.run(args.host, args.port, args.debug)

    return 0

if __name__ == "__main__":
    sys.exit(main())
