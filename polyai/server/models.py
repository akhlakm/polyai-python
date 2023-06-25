import os
import gc
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
        t1 = log.trace("Unloading current model.")
        polyai.server.model = None
        polyai.server.token = None
        polyai.server.pipeline = None
        gc.collect()
        torch.cuda.empty_cache()
        t1.done("Unloaded current model.")


def init_gptq_model(modelname, groupsize=-1, fused_mlp=False, use_fast=False):
    """
    Load a pretrained 4 bit GPTQ model using GPTQ-for-LLaMa.
    Arguments:
        modelname str : path to model safetensor file
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
    polyai.server.modelName = os.path.basename(modelname).split(".")[0]

    t1.done("Model loaded: {}", modelname)
    vram_usage()


def init_hf_bert(model_path):
    """
    Load a pretrained HF bert model.
    Arguments:
        model_path str : path to model directory    
    """
    from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

    unload_model()
    t1 = log.trace("Loading BERT model: {}", model_path)
    vram_usage()

    polyai.server.modelName = os.path.basename(model_path).split(".")[0]

    polyai.server.token = AutoTokenizer.from_pretrained(model_path, model_max_length=512)
    polyai.server.model = AutoModelForTokenClassification.from_pretrained(model_path)

    # Load model and tokenizer
    polyai.server.pipeline = pipeline(task="ner",
                                      model=polyai.server.model,
                                      tokenizer=polyai.server.token,
                                      aggregation_strategy="simple",
                                      device="cuda:0")

    t1.done("Model loaded: {}", model_path)
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


import spacy
from collections import namedtuple
nlp = spacy.load("en_core_web_sm")

def _ner_feed(seq_pred, text) -> list[namedtuple]:
    """ Convert outputs of the NER to a form usable by record extraction
        seq_pred: List of dictionaries
        text: str, text fed to sequence classification model
    """
    doc = nlp(text)
    token_label = namedtuple('token_label', ["text", "label"])
    if len(seq_pred) == 0:
        # If no NER could be reconginzed, the prediction list would be empty.
        return [token_label(doc[i].text, 'O') for i in range(len(doc))]

    seq_index = 0
    text_len = len(text)
    seq_len = len(seq_pred)
    len_doc = len(doc)
    token = ''
    token_labels = []
    start_index = seq_pred[seq_index]["start"]
    end_index = seq_pred[seq_index]["end"]
    i = 0
    char_index = -1

    while i < len_doc:
        token = doc[i].text
        if char_index+1 >= start_index and seq_index < seq_len:
            # Continue loop till end_index or end of word
            # increment index and values
            current_label = seq_pred[seq_index]["entity_group"]
            while char_index < end_index-1:
                token_labels.append(token_label(token, current_label))
                char_index += len(token)
                if char_index < text_len-1 and text[char_index+1] == ' ': char_index+=1
                i += 1
                if i < len_doc: token=doc[i].text
            seq_index += 1
            if seq_index < seq_len:
                start_index = seq_pred[seq_index]["start"]
                end_index = seq_pred[seq_index]["end"]
        else:
            token_labels.append(token_label(token, 'O'))
            i += 1
            char_index += len(token)
            if char_index < text_len-1 and text[char_index+1] == ' ':
                char_index += 1
    
    return token_labels 

def get_bert_ner(text):
    """
    Perform NER using the loaded BERT model on the given text.
    Returns:
        List of generated NER tags as a dict format,
        Total time elapsed in miliseconds.
    """
    t1 = log.trace("Getting NER for: {}", text)
    ner_output = polyai.server.pipeline(text)
    print(ner_output)
    ner_tuples = _ner_feed(ner_output, text)
    t1.done("NER processed")
    return [tup._asdict() for tup in ner_tuples], round(1000 * t1.elapsed())
