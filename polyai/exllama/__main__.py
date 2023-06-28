"""
ExLlama script to perform tests inside a docker environment.

"""

import sys
from . import load_model, generate, simple

if __name__ == "__main__":
    model = "/home/user/models/TheBloke_stable-vicuna-13B-GPTQ/stable-vicuna-13B-GPTQ-4bit.compat.no-act-order.safetensors"
    load_model(model)
    print("Load OK")

    # print(simple("### Human: Hello there!\n### Assistant:"))

    print("Streaming ...")
    for resp in generate("""
    A chat between two friends where one of them is a bit too eager to share their opinions and experiences.
    ### Human: What is the meaning of life?
    ### Assistant:""", ("### Human", "### Assistant")):
        print(resp, end="")
        sys.stdout.flush()
        # time.sleep(0.05)

