import polyai.server.api.api_v2.blocking as blocking_api
import polyai.server.api.api_v2.streaming as streaming_api

# API version
__version__ = "2.0"

def run(blocking_port, streaming_port, listen=False):
    blocking_api.start_server(blocking_port, listen)
    streaming_api.start_server(streaming_port, listen)
