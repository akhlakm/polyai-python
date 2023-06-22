#!/usr/bin/env python

# The implementation is same as openai !!
import polyai.api as polyai

import os
from dotenv import load_dotenv
from polyai.api.util import setup_ssh_tunnel

load_dotenv()       # load env variables
setup_ssh_tunnel()  # enable this if needed

polyai.api_key = os.environ.get("POLYAI_API_KEY")

resp = polyai.ChatCompletion.create(
    model="gpt-3.5-turbo", # currently ignored by the server.
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"},
        {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
        {"role": "user", "content": "Where was it played?"}
    ],
    temperature = 0.1,
    max_length = 512,
)

print(resp, end="\n\n") # server response object, also same as openai
print(resp['choices'][0]['message']['content']) # actual response
