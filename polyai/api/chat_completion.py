import time
import pylogg
from polyai import engine, error

log = pylogg.New("polyai")


class ChatCompletion(engine.APIEngine):
    __object__ = "chat.completions"

    @classmethod
    def create(cls, *args, **kwargs):
        start = time.time()
        timeout = kwargs.pop("timeout", None)

        while True:
            try:
                return super().create(*args, **kwargs)
                
            except error.TryAgain as e:
                if timeout is not None and time.time() > start + timeout:
                    raise

                log.note("Waiting for model to warm up: error={}", e)


class Completion(ChatCompletion):
    __object__ = "completions"


def dummy_response():
    return {
        'id': 'chatcmpl-6p9XYPYSTTRi0xEviKjjilqrWU2Ve',
        'object': 'chat.completion',
        'created': 1677649420,
        'model': 'gpt-3.5-turbo',
        'usage': {
            'prompt_tokens': 56,
            'completion_tokens': 31,
            'total_tokens': 87
        },
        'choices': [
            {
                'message': {
                    'role': 'assistant',
                    'content': 'The 2020 World Series was played in Arlington, Texas at the Globe Life Field, which was the new home stadium for the Texas Rangers.'
                },
                'finish_reason': 'stop',
                'index': 0
            }
        ]
    }
