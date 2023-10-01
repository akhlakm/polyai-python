import time
import pylogg
from polyai.api import engine, error

log = pylogg.New("polyai")


class TextEmbedding(engine.APIEngine):
    __object__ = 'text.embedding'

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
