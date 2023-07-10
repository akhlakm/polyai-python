import os
import sys
import dotenv
import argparse
import pylogg as log

from polyai import __version__
from polyai.server import loader
from polyai.server.api import api_v1


def parse_arguments():
    parser = argparse.ArgumentParser(description="PolyAI Server (v%s)" %__version__)

    parser.add_argument("cmd", help="server")

    parser.add_argument("--host", default="0.0.0.0", help="server host IP address")
    parser.add_argument("--port", default=8080, type=int, help="server port")
    parser.add_argument("--model", default=None, help="LLM model safetensors or pt to load")
    parser.add_argument("--lora", default=None, help="Path to LoRA directory to load")
    parser.add_argument("--bert", default=None, help="Path to BERT model directory")
    parser.add_argument("--vram", default=None, help="Comma seperated max VRAM usage for the GPUs")

    parser.add_argument("--debug", default=True, action='store_true',
                        help="enable debugging")

    args = parser.parse_args()

    return args


def model_path(env_var, is_file=False) -> str | None:
    """ Check if path exists from env variable. """
    path = os.getenv(env_var)
    if path is None or len(path.strip()) == 0:
        return None
    path = os.path.join("models", path)
    if is_file:
        if os.path.isfile(path):
            return path
    else:
        if os.path.isdir(path):
            return path
    return None


def main():
    args = parse_arguments()
    if args.debug:
        log.setLevel(log.DEBUG)
        log.setMaxLength(2000)
        log.setFile(open("polyai.log", "a+"))
        log.setFileTimes(show=False)
    else:
        log.setLevel(log.INFO)
        log.setMaxLength(1000)
        log.setFile(open("polyai.log", "a+"))
        log.setConsoleTimes(show=True)

    if not dotenv.load_dotenv():
        raise RuntimeError("ENV not loaded")

    if args.cmd == "server":
        if args.model is None:
            args.model = model_path("POLYAI_MODEL_PATH", is_file=True)

        if args.lora is None:
            args.lora = model_path("POLYAI_LORA_DIR")

        if args.bert is None:
            args.bert = model_path("POLYAI_BERT_DIR")


        # Load the models into memory
        if args.model is not None:
            exllama = loader.init_exllama(args)
            exllama.load_model(args.model)
            if args.lora:
                exllama.add_lora(args.lora)
        else:
            log.error("No valid modelpath specified. Models can specified using the --model argument.")
            log.error("Alternatively, set the POLYAI_MODEL_PATH relative to ./models/ directory.")

        if args.bert is not None:
            bert = loader.init_bert(args)
            bert.load_model(args.bert)
        else:
            log.warning("No valid BERT directory specified. BERT can specified using the --bert argument.")
            log.warning("Alternatively, set the POLYAI_BERT_DIR relative to ./models/ directory.")


        # Start the API servers.
        log.info("Running server on {}:{}", args.host, args.port)
        api_v1.run(args.host, args.port, debug=False)


    log.close()
    return 0

if __name__ == "__main__":
    sys.exit(main())
