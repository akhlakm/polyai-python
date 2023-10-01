"""
Entry point for polyai server.
"""

import sys
import argparse
import pylogg as log

from polyai import __version__, sett

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="PolyAI Server (v%s)" %__version__)
    
    parser.add_argument("cmd", choices=['server', 'examples'])
    parser.add_argument(
        "--model", default=None,
        help="LLM model safetensors or pt to load")
    parser.add_argument(
        "--ssl", default=None, action="store_true",
        help="Use https for requests")
    parser.add_argument(
        "--vram", default=None,
        help="Comma seperated max VRAM usage for the GPUs")
    parser.add_argument(
        "--log", default=None, type=int,
        help="Log level. Higher is more verbose.")
    parser.add_argument(
        "--debug", default=None, action='store_true',
        help="Enable debugging")

    args = parser.parse_args()
    return args


def server():
    from polyai.server import loader
    from polyai.server import endpoints

    _start = False

    if sett.Model.model_file_path:
        exllama = loader.init_exllama(
            sett.TextGen.user_fmt,
            sett.TextGen.bot_fmt,
            sett.TextGen.instruction_fmt,
            sett.Model.vram_config, sett.TextGen.context_length)

        exllama.load_model(sett.Model.model_file_path)
        _start = True

        if sett.Model.lora_file_path:
            exllama.add_lora(sett.Model.lora_file_path)
    else:
        log.error("No language model file specified.")

    if sett.Model.bert_file_path:
        bert = loader.init_bert(sett.Model.bert_device)
        bert.load_model(sett.Model.bert_file_path)
        _start = True
    else:
        log.warn("No BERT model file specified.")

    # Start the API servers, v1 is blocking, so run it last.
    if _start:
        endpoints.run(
            polyai_port=sett.Server.api_endpoint_port,
            streaming_port=sett.Server.stream_endpoint_port,
            listen=sett.Server.listen_all,
            ssl=sett.Server.use_ssl,
            debug=sett.Server.debug
        )


def main() -> int:
    if not sett.load_server_settings():
        sett.save_server_settings()
        print("Please update the new settings file and retry.")
        return 1
    else:
        sett.save_server_settings()
    
    args = parse_arguments()

    # Override settings from args.
    if args.log is not None:
        sett.Server.log_level = args.log
    if args.ssl is not None:
        sett.Server.use_ssl = args.ssl
    if args.debug is not None:
        sett.Server.debug = args.debug
    if args.model is not None:
        sett.Model.model_file_path = args.model
    if args.vram is not None:
        sett.Model.vram_config = args.vram

    t1 = log.init(sett.Server.log_level, output_directory=".",
             logfile_name=sett.Server.log_file_name,
             append_to_logfile=sett.Server.log_append)

    if args.cmd == "server":
        server()
    else:
        ValueError(args.cmd)

    t1.note("All done.")
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
