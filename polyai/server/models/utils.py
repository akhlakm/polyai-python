import gc
import torch

import pylogg
import polyai.server

log = pylogg.New("llm")


def spacy_nlp(doc : str):
    """ Return the spacy NLP instance of a text. """
    import spacy
    nlp = spacy.load("en_core_web_sm")
    return nlp(doc)


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
