import os
from collections import namedtuple

import pylogg
from . import utils

log = pylogg.New("llm")

class BERT:
    """ Global variables """
    modelName : str = None
    pipeline = None


def init_hf_bert(model_path):
    """
    Load a pretrained HF bert model.
    Arguments:
        model_path str : path to model directory    
    """
    from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

    t1 = log.trace("Loading BERT model: {}", model_path)
    utils.vram_usage()

    BERT.modelName = os.path.basename(model_path).split(".")[0]

    tokenizer = AutoTokenizer.from_pretrained(model_path, model_max_length=512)
    model = AutoModelForTokenClassification.from_pretrained(model_path)

    # Load model and tokenizer
    BERT.pipeline = pipeline(task="ner",
                            model=model,
                            tokenizer=tokenizer,
                            aggregation_strategy="simple",
                            device="cuda:0")

    t1.done("Model loaded: {}", BERT.modelName)
    utils.vram_usage()


def get_bert_ner(text):
    """
    Perform NER using the loaded BERT model on the given text.
    Returns:
        Model name,
        List of generated NER tags as a dict format,
        Total time elapsed in miliseconds.
    """
    t1 = log.trace("Getting NER for: {}", text)
    ner_output = BERT.pipeline(text)
    print(ner_output)
    ner_tuples = _ner_feed(ner_output, text)
    t1.done("NER processed")

    return (
        BERT.modelName,
        [tup._asdict() for tup in ner_tuples],
        round(1000 * t1.elapsed())
    )


def _ner_feed(seq_pred, text) -> list[namedtuple]:
    """ Convert outputs of the NER to a form usable by record extraction
        seq_pred: List of dictionaries
        text: str, text fed to sequence classification model
    """
    doc = utils.spacy_nlp(text)
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
