import ssl
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pylogg
import polyai.server.state as state

# Set log prefix
log = pylogg.New("api/v2")

PATH = "/api/v1"

def get_model_info():
    return {
        'model_name': state.LLM.model_name(),
        'lora_names': [state.LLM._lora_name],
        # dump
        'shared.settings': {},
        'shared.args': [],
    }


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        log.trace("Get request: {}", self.path)
        # Send currently loaded model name
        if self.path == PATH+'/model':
            self.send_response(200)
            self.end_headers()
            response = json.dumps({
                'result': state.LLM.model_name()
            })
            self.wfile.write(response.encode('utf-8'))
        else:
            self.send_error(404)

    def do_POST(self):
        log.trace("Post request: {}", self.path)
        content_length = int(self.headers['Content-Length'])
        body = json.loads(self.rfile.read(content_length).decode('utf-8'))

        if self.path == PATH+'/generate':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()

            prompt = body['prompt']
            output = state.LLM.generate(prompt, body)
            model, reply_list, ptok, ctok, dt = output

            response = json.dumps({
                'results': [{
                    'text': "\n".join(reply_list)
                }]
            })

            self.wfile.write(response.encode('utf-8'))

        elif self.path == PATH+'/chat':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()

            user_input = body['user_input']
            body['stream'] = False

            # Chat reply
            # regenerate = body.get('regenerate', False)
            # _continue = body.get('_continue', False)
            # generator = generate_chat_reply(
            #     user_input, generate_params, regenerate=regenerate, _continue=_continue, loading_message=False)
            # answer = generate_params['history']
            log.warn("Chat generation requested. Not fully supported.")

            output = state.LLM.generate(prompt, body)
            model, reply_list, ptok, ctok, dt = output

            response = json.dumps({
                'results': [{
                    'history': "\n".join(reply_list)
                }]
            })

            self.wfile.write(response.encode('utf-8'))

        elif self.path == PATH+'/stop-stream':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()

            log.trace("Stop generation requested.")
            state.LLM.stop_generation()

            response = json.dumps({
                'results': 'success'
            })

            self.wfile.write(response.encode('utf-8'))

        elif self.path == PATH+'/model':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()

            # by default return the same as the GET interface
            result = state.LLM.model_name()

            # Actions: info, load, list, unload
            action = body.get('action', '')

            if action == 'load':
                model_name = body['model_name']
                args = body.get('args', {})
                print('args', args)

                response = json.dumps({'error': {'message': 'not allowed'}})
                self.wfile.write(response.encode('utf-8'))
                raise NotImplementedError("Model load request")

            elif action == 'unload':
                response = json.dumps({'error': {'message': 'not allowed'}})
                self.wfile.write(response.encode('utf-8'))
                raise NotImplementedError("Model unload request")

            elif action == 'list':
                log.trace("Model list requested.")
                result = [state.LLM.model_name()]

            elif action == 'info':
                log.trace("Model info requested.")
                result = get_model_info()

            response = json.dumps({
                'result': result,
            })

            self.wfile.write(response.encode('utf-8'))

        elif self.path == PATH+'/token-count':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()

            log.trace("Token count requested: {}", body['prompt'])
            tokens = state.LLM.encode(body['prompt'])[0]
            response = json.dumps({
                'results': [{
                    'tokens': len(tokens)
                }]
            })

            self.wfile.write(response.encode('utf-8'))
        else:
            self.send_error(404)

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', '*')
        self.send_header('Access-Control-Allow-Headers', '*')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        super().end_headers()


def _run_server(port: int, listen : bool = False, secure : bool = True):
    address = '0.0.0.0' if listen else '127.0.0.1'
    protocol = 'https' if secure else 'http'

    server = ThreadingHTTPServer((address, port), Handler)

    if secure:
        server.socket = ssl.wrap_socket(server.socket,
                                        server_side=True,
                                        certfile="keys/ssl.crt",
                                        keyfile="keys/ssl.key",
                                        ssl_version=ssl.PROTOCOL_TLS)

    log.info(f'Running blocking server at {protocol}://{address}:{port}{PATH}')
    server.serve_forever()


def start_server(port: int, listen: bool = False):
    Thread(target=_run_server, args=[port, listen], daemon=False).start()
