# POLYAI
A drop-in replacement for OpenAI API for local LLMs

## Installation
Clone this repo and install with `pip`.

```sh
git clone https://github.com/akhlakm/polyai-python.git
cd polyai-python
pip install -e .
```

## Usage
```python
import os
from dotenv import load_dotenv

import polyai.api as polyai

load_dotenv()       # load .env variables
polyai.api_key = os.environ.get("POLYAI_API_KEY")

polyai.create_ssh_tunnel()  # use this if needed

# The implementation is same as openai.
resp = polyai.ChatCompletion.create(
    model="polyai",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"},
        {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
        {"role": "user", "content": "Where was it played?"}
    ],
    temperature = 0.1,
    max_length = 512,
)

print(resp, end="\n\n") # response object, also same as openai
print(resp['choices'][0]['message']['content']) # response text
```

See the [tests directory](/tests) for more example scripts.

## Todos
- [X] Move arguments parsing to the `__main__.py` file.
- [X] Add textgen ui api endpoint as v2.
- [ ] Implement Langchain with polyai.
- [ ] Add QLora training module.
