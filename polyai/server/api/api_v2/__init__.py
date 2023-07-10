import polyai.server.api.api_v2.blocking as blocking_api
import polyai.server.api.api_v2.streaming as streaming_api

def run(blocking_port, streaming_port):
    blocking_api.start_server(blocking_port)
    streaming_api.start_server(streaming_port)
