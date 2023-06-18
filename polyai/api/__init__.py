import os
from .chat_completion import Completion
from .chat_completion import ChatCompletion

__version__ = "2023.06.18r1"
__author__ = "Akhlak Mahmood"

api_key = os.environ.get("POLYAI_API_KEY")
api_key_path = os.environ.get("POLYAI_API_KEY_PATH")
api_base = os.environ.get("POLYAI_API_BASE", "http://localhost:8080/v1")

session = None
app_info = None

