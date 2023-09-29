import time
import os, sys

import pylogg
import polyai.server
from polyai.server.loader import utils

log = pylogg.New("llm")


def init_gptq_model(modelname, groupsize=-1, fused_mlp=False, use_fast=False):
    """
    Load a pretrained 4 bit GPTQ model using GPTQ-for-LLaMa.
    Arguments:
        modelname str : path to model safetensor file
        groupsize int : GPTQ group size, default -1.
        use_fast bool : Use fast tokenizer.
        fused_mlp bool : Use fused mlp or not, use false if model loading fails.
    """
    # import gptq
    gptq_repo = os.environ.get("POLYAI_GPTQ_DIR", "GPTQ-for-LLaMa")
    sys.path.append(os.path.abspath(gptq_repo))
    try:
        import llama_inference as gptq
    except Exception as err:
        log.critical("Failed to import GPTQ: {} ", err)
        log.note("Please clone "
                "https://github.com/qwopqwop200/GPTQ-for-LLaMa "
                "to current working directory.")

    utils.unload_model()
    t1 = log.trace("Loading GPTQ model: {}", modelname)
    utils.vram_usage()

    modeldir = os.path.dirname(modelname)
    polyai.server.model = gptq.load_quant(modeldir, modelname, 4, groupsize, fused_mlp=fused_mlp)
    polyai.server.model.to(gptq.DEV)
    polyai.server.token = gptq.AutoTokenizer.from_pretrained(modeldir, use_fast=use_fast)
    polyai.server.modelName = os.path.basename(modelname).split(".")[0]

    t1.done("Model loaded: {}", modelname)
    utils.vram_usage()


def get_gptq_response(prompt, maxlen=512, top_p=0.95, temp=0.1, minlen=10, **kwargs):
    """
    Given a prompt message, generate model response.
    
    Returns:
        List of generated responses,
        Total input tokens,
        Total completion tokens,
        Total time elapsed in miliseconds.
    """
    import llama_inference as gptq
    t1 = log.trace("Getting response for: {}", prompt)

    start = time.time()
    input_ids = polyai.server.token.encode(prompt, return_tensors="pt").to(gptq.DEV)

    prompt_tok = 0
    for i in input_ids:
        prompt_tok += len(i)

    with gptq.torch.no_grad():
        generated_ids = polyai.server.model.generate(input_ids,
            do_sample=True, min_length=minlen,
            max_length=maxlen, top_p=top_p,
            temperature=temp)

    compl_tok = 0
    outputs = []
    for gen in generated_ids:
        compl_tok += len(gen)
        tokens = [el.item() for el in gen]
        outputs.append(polyai.server.token.decode(tokens))

    compl_tok -= prompt_tok
    delta = time.time() - start

    t1.done("Response: {}", outputs)
    return outputs, prompt_tok, compl_tok, round(1000 * delta)

