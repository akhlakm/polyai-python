#!/usr/bin/env python

"""
Example usages of helpers to instruct the LLM
with two-shot learning.

"""

from dotenv import load_dotenv
from polyai.api.util import create_ssh_tunnel
from polyai.api.helpers import (
    generation_time, tok_per_sec,
    instruct_prompt, model_reply
)

# Load the env variables.
load_dotenv()

# Uncomment this to use ssh tunnel. Make sure to
# add hostname, username and password in the .env file.
create_ssh_tunnel()

print("Sending api request.")
resp, ptok, ctok, req = instruct_prompt(
    model="polyai",
    instruction="Respond to user messages in a helpful way.",
    prompt="Can you write me a poem about polymers?",
    shots = [
        ("hello", "hi there"),
        ("How are you?", "I'm doing fantastic, how about you?")
    ]
)

print("\nResponse stats:")
print("Tokens (prompt completion) =", ptok, ctok)
print("Generation time =", generation_time(resp)/1000, "sec")
print("Tokens per second =", tok_per_sec(resp), "\n")
print(model_reply(resp))
