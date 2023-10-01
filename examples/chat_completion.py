#!/usr/bin/env python
"""
Example use of few-shot chat completion of polyai api.
The implementation is same as openai.

"""

import polyai.api as polyai
import polyai.sett as sett

try:
    sett.load_api_settings()
except:
    pass
sett.save_api_settings()

polyai.create_ssh_tunnel()  # if needed

polyai.api_key = sett.API.polyai_api_key
polyai.api_base = sett.API.polyai_api_base

resp = polyai.ChatCompletion.create(
    model="polyai", # currently ignored by the server.
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
