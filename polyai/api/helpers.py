import os
import polyai.api


def instruct_prompt(model, instruction, prompt, shots = [], **kwargs):
    """
    Use polyai API to query a model with given instruction and prompt.
    Optionally provide (user, asst) shots as a list of tuples.

    Returns:
        Response dict,
        No of prompt tokens,
        No of completion tokens.
    """

    polyai.api.api_key = os.getenv("POLYAI_API_KEY")
    
    # Construct the instruct payload with optional shots
    # No of shots = len(payload) - 2
    payload = [ {"role": "system", "content": instruction} ]
    for shot in shots:
        user = shot[0]
        asst = shot[1]
        payload.append({"role": "user", "content": user})
        payload.append({"role": "assistant", "content": asst})
    payload.append({"role": "user", "content": prompt})

    # Make API request
    resp = polyai.api.ChatCompletion.create(model=model, messages=payload, **kwargs)

    p_tok = resp['usage']['prompt_tokens']
    c_tok = resp['usage']['completion_tokens']

    return resp, p_tok, c_tok


def model_reply(respObj, i = 0):
    """ Extract the ith model reply from the response json. """
    return respObj['choices'][i]['message']['content']