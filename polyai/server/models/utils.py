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
        current usage, total vram, free vram
    """
    free = 0
    total = 0
    devices = torch.cuda.device_count()
    for i in range(devices):
        mem = torch.cuda.mem_get_info(i)
        free += mem[0]/1024/1024/1024
        total += mem[1]/1024/1024/1024
    log.info("VRAM usage, %d GPUs: %0.4f GB / %0.4f GB (%0.4f GB free)" %(devices, total-free, total, free))
    return total - free, total, free
