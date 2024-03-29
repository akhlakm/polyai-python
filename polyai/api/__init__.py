import os
from . import error
from .chat_completion import Completion
from .chat_completion import ChatCompletion
from .bert_ner import BERTNER
from .embedding import TextEmbedding
from .util import create_ssh_tunnel

api_key = os.environ.get("POLYAI_API_KEY")
api_key_path = os.environ.get("POLYAI_API_KEY_PATH")
api_base = os.environ.get("POLYAI_API_BASE", "http://localhost:8001/polyai/")

session = None
app_info = None

del os, util
