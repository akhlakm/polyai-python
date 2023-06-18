import os
import sys

import pylogg
import polyai.server

log = pylogg.New("polyai")


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


def unload_model():
    if polyai.server.model is not None:
        import gc, torch
        t1 = log.trace("Unloading current model.")
        polyai.server.model = None
        polyai.server.token = None
        gc.collect()
        torch.cuda.empty_cache()
        t1.done("Unloaded current model.")


def init_gptq_model(modelname, use_fast=False):
    unload_model()
    t1 = log.trace("Loading GPTQ model: {}", modelname)

    modeldir = os.path.dirname(modelname)
    polyai.server.model = gptq.load_quant(modeldir, modelname, 4, 128, fused_mlp=True)
    polyai.server.model.to(gptq.DEV)
    polyai.server.token = gptq.AutoTokenizer.from_pretrained(modeldir, use_fast=use_fast)

    t1.done("Model loaded: {}", modelname)


def init_hf_model(pipeline_type, modelname):
    from transformers import pipeline
    unload_model()
    t1 = log.trace("Loading HuggingFace model: {}", modelname)
    polyai.server.model = pipeline(pipeline_type, model=modelname)
    t1.done("Model loaded: {}", modelname)


def get_gptq_response(prompt, maxlen=512, top_p=0.95, temp=0.8, **kwargs):
    t1 = log.trace("Getting response for: {}", prompt)
    input_ids = polyai.server.token.encode(prompt, return_tensors="pt").to(gptq.DEV)

    with gptq.torch.no_grad():
        generated_ids = polyai.server.model.generate(input_ids,
            do_sample=True, min_length=10,
            max_length=maxlen, top_p=top_p,
            temperature=temp)

    output = polyai.server.token.decode([el.item() for el in generated_ids[0]])
    t1.done("Response: {}", output)
    return output
