import time

def make_response_dict(idstr : str, object : str, model : str, dt : int,
               prompt_tok : int, compl_tok : int, choices : list = [],
               ner : list = []):
    
    """ Construct and return an OPENAI like response dict for json payload. """
    
    if len(ner) == 0 and len(choices) == 0:
        raise ValueError("Either ner or choices need to be provided")
    
    # set indices
    for i, ch in enumerate(choices):
        choices[i]['index'] = i

    for i, ch in enumerate(ner):
        ner[i]['index'] = i

    return {
        'id': idstr,
        'object': object,
        'created': time.strftime("%a, %b %d %Y %X"),
        'model': model,
        'usage': {
            'prompt_tokens': prompt_tok,
            'completion_tokens': compl_tok,
            'total_tokens': prompt_tok + compl_tok
        },
        'elapsed_msec': dt,
        'choices': choices,
        'ner_tags': ner,
    }


def make_choice_dict(response, finish_reason):
    # openai like response message object
    return {
        'message': {
            'role': 'assistant',
            'content': response,
        },
        'finish_reason': finish_reason
    }
