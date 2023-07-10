import os
from collections import namedtuple
import pylogg
import polyai.server.state as state

log = pylogg.New("bert")


class BERTModel:
    def __init__(self, device = None) -> None:
        import spacy
        self.device = device if device else 'cuda:0'
        self.nlp = spacy.load("en_core_web_sm")

    def print_vram_usage(self):
        log.info(
            "VRAM usage, %d GPUs: %0.4f GB / %0.4f GB (%0.4f GB free)"
            %state.Server.vram_usage())

    def load_model(self, model_dir):
        # Notify user
        t1 = log.trace("Loading BERT model: {}", model_dir)
        self.print_vram_usage()

        from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

        state.BERT._model_name = os.path.basename(model_dir).split(".")[0]
        tokenizer = AutoTokenizer.from_pretrained(model_dir, model_max_length=512)
        model = AutoModelForTokenClassification.from_pretrained(model_dir)

        # Load model and tokenizer
        state.BERT._pipeline = pipeline(task="ner",
                                 model=model,
                                 tokenizer=tokenizer,
                                 aggregation_strategy="simple",
                                 device=self.device)

        t1.done("Model loaded: {}", state.BERT.model_name())
        self.print_vram_usage()

    def ner_tags(self, text):
        t1 = log.trace("Getting NER for: {}", text)
        ner_output = state.BERT._pipeline(text)
        ner_tuples = _ner_feed(ner_output, text)
        t1.done("NER processed: {}", ner_output)

        return (
            state.BERT.model_name(),
            [tup._asdict() for tup in ner_tuples],
            round(1000 * t1.elapsed())
        )


    def _ner_feed(self, seq_pred, text) -> list[namedtuple]:
        """ Convert outputs of the NER to a form usable by record extraction
            seq_pred: List of dictionaries
            text: str, text fed to sequence classification model
        """
        doc = self.nlp(text)
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


def init_bert(args):
    if args.bert_device == 'cuda':
        args.bert_device = 'cuda:0'

    state.BERT._loader = BERTModel(args.bert_device)
    return state.BERT._loader
