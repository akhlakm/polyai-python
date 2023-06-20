import time
import pylogg
from polyai.api import engine, error

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
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "Himalayas, Nepal, India",
                    "role": "assistant"
                }
            }
        ],
        "created": "Tue, Jun 20 2023 02:53:53",
        "elapsed_msec": 2061,
        "id": "chcmpl-1687229633485",
        "model": "Wizard-Vicuna-13B-Uncensored-GPTQ-4bit-128g.compat.no-act-order.safetensors",
        "object": "chat.completions",
        "usage": {
            "completion_tokens": 10,
            "prompt_tokens": 70,
            "total_tokens": 80
        }
    }
