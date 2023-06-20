from dotenv import load_dotenv
from polyai import api
from polyai.api.helpers import generation_time, tok_per_sec

load_dotenv()

r, ptok, ctok = api.instruct_prompt("",
    "Respond to user messages in a helpful way.",
    "Me too. Can you write me a poem about polymers?",
    shots = [
        ("hello", "hi there"),
        ("How are you?", "I'm doing fantastic, how about you?")
    ]
)

print("Tokens =", ptok, ctok)
print("Tokens per second =", tok_per_sec(r))
print(api.model_reply(r))
