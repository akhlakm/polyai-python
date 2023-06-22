import os
from .chat_completion import Completion
from .chat_completion import ChatCompletion
from .helpers import instruct_prompt, model_reply
from .util import ssh_tunnel

api_key = os.environ.get("POLYAI_API_KEY")
api_key_path = os.environ.get("POLYAI_API_KEY_PATH")
api_base = os.environ.get("POLYAI_API_BASE", "http://localhost:8080/api/")

session = None
app_info = None

del helpers, os
