import os
import sys
import time
import torch

import pylogg
import polyai.server

log = pylogg.New("llm")


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


def vram_usage():
    """ Log and return the vram usage.
    Returns:
        current usage, total vram
    """
    mem = torch.cuda.mem_get_info()
    o = mem[0]/1024/1024/1024
    t = mem[1]/1024/1024/1024
    log.info("VRAM usage: %0.4f GB / %0.4f GB" %(o, t))
    return o, t


def unload_model():
    """ Attempt to clear the momery by unloading any
    existing model.
    """
    if polyai.server.model is not None:
        import gc, torch
        t1 = log.trace("Unloading current model.")
        polyai.server.model = None
        polyai.server.token = None
        gc.collect()
        torch.cuda.empty_cache()
        t1.done("Unloaded current model.")


def init_gptq_model(modelname, groupsize=-1, fused_mlp=False, use_fast=False):
    """
    Load a pretrained 4 bit GPTQ model using GPTQ-for-LLaMa.
    Arguments:
        groupsize int : GPTQ group size, default -1.
        use_fast bool : Use fast tokenizer.
        fused_mlp bool : Use fused mlp or not, use false if model loading fails.
    """
    unload_model()
    t1 = log.trace("Loading GPTQ model: {}", modelname)
    vram_usage()

    modeldir = os.path.dirname(modelname)
    polyai.server.model = gptq.load_quant(modeldir, modelname, 4, groupsize, fused_mlp=fused_mlp)
    polyai.server.model.to(gptq.DEV)
    polyai.server.token = gptq.AutoTokenizer.from_pretrained(modeldir, use_fast=use_fast)
    polyai.server.modelName = os.path.basename(modelname)

    t1.done("Model loaded: {}", modelname)
    vram_usage()


def init_hf_model(pipeline_type, modelname):
    from transformers import pipeline
    unload_model()
    t1 = log.trace("Loading HuggingFace model: {}", modelname)
    polyai.server.model = pipeline(pipeline_type, model=modelname)
    t1.done("Model loaded: {}", modelname)


def get_gptq_response(prompt, maxlen=512, top_p=0.95, temp=0.1, minlen=10, **kwargs):
    """
    Given a prompt message, generate model response.
    
    Returns:
        List of generated responses,
        Total input tokens,
        Total completion tokens,
        Total time elapsed in miliseconds.
    """
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
