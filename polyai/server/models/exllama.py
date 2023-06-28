import os, sys
import argparse
import torch

import pylogg
log = pylogg.New("llm")

from polyai.exllama import model_init
from polyai.exllama.model import ExLlama, ExLlamaCache
from polyai.exllama.lora import ExLlamaLora
from polyai.exllama.tokenizer import ExLlamaTokenizer
from polyai.exllama.generator import ExLlamaGenerator

from polyai.server.models import utils


class LLM:
    """ Global variables """
    modelName : str = None
    loraName : str = None
    model : ExLlama = None
    tokenizer : ExLlamaTokenizer = None
    cache : ExLlamaCache = None
    lora : ExLlamaLora = None
    args : argparse.Namespace = None
    system : str = ""
    user : str = os.getenv("POLYAI_USER_FMT", "USER:")
    bot  : str = os.getenv("POLYAI_BOT_FMT", "ASSISTANT:")


def init_exllama_model(modelpath, lora_dir = None):
    """
    Load a pretrained 4 bit GPTQ model using ExLlama.
    Arguments:
        modelname str : path to model safetensor file
        lora_dir  str : path to lora directory
    """

    t1 = log.trace("Loading ExLlama model: {}", modelpath)
    if lora_dir:
        log.note("Using LoRA from: {}", lora_dir)
    utils.vram_usage()

    torch.set_grad_enabled(False)
    torch.cuda._lazy_init()

    # Default exllama options
    parser = argparse.ArgumentParser(description = "ExLlama")
    model_init.add_args(parser)
    args = parser.parse_args(args=[])

    # Overrides/settings
    args.directory = os.path.dirname(modelpath)

    # Post process the arguments
    model_init.get_model_files(args)

    # Load the model, tokenizer and lora if any
    config = model_init.make_config(args)
    model = ExLlama(config)
    cache = ExLlamaCache(model)
    tokenizer = ExLlamaTokenizer(args.tokenizer)
    lora = load_exllama_lora(model, lora_dir)

    LLM.args = args
    LLM.model = model
    LLM.tokenizer = tokenizer
    LLM.cache = cache
    LLM.lora = lora
    LLM.modelName = os.path.basename(modelpath).split(".")[0]

    model_init.print_stats(LLM.model)
    t1.done("Model loaded: {}", LLM.modelName)
    utils.vram_usage()


def load_exllama_lora(model, loradir = None) -> ExLlamaLora:
    """ Load an exllama lora given it's directory. """

    if loradir is None:
        return None
    
    lora_config = os.path.join(loradir, "adapter_config.json")
    lora_bin = os.path.join(loradir, "adapter_model.bin")

    lora = ExLlamaLora(model, lora_config, lora_bin)
    if lora.bias_ignored:
        print(f" !! Warning: LoRA zero bias ignored")

    return lora


def get_exllama_response(prompt, stream = False, **kwargs):
    """
    Given a prompt message, generate model response.
    
    Returns:
        Name of the model,
        List of generated responses,
        Total input tokens,
        Total completion tokens,
        Total time elapsed in miliseconds.
    """

    t1 = log.trace("Getting LLM response for: {}", prompt)

    generator = ExLlamaGenerator(LLM.model, LLM.tokenizer, LLM.cache)
    generator.settings = ExLlamaGenerator.Settings()
    generator.settings.temperature = float(kwargs.get("temperature") or 0.1)
    generator.settings.top_k = int(kwargs.get("top_k") or 20)
    generator.settings.top_p = float(kwargs.get("top_p") or 0.95)
    generator.settings.min_p = float(kwargs.get("min_p") or 0.0)
    generator.settings.token_repetition_penalty_max = float(kwargs.get("repetition_penalty") or 1.0)
    generator.settings.token_repetition_penalty_sustain = int(kwargs.get("repetition_penalty_sustain") or 256)
    generator.settings.token_repetition_penalty_decay = generator.settings.token_repetition_penalty_sustain // 2
    generator.settings.beams = int(kwargs.get("beams") or 1)
    generator.settings.beam_length = int(kwargs.get("beam_length") or 1)
    generator.lora = LLM.lora

    max_tokens = int(kwargs.get("max_tokens") or 512)
    break_on_newline = False

    participants = [LLM.user, LLM.bot]

    # Prepare stop conditions
    stop_conditions = []
    newline_token = torch.Tensor([[LLM.tokenizer.newline_token_id]]).long()

    if break_on_newline:
        # Stop generation on newline character.
        stop_conditions.append((newline_token, "\n"))
    else:
        # Stop generation if a newline followed by a participant is generated.
        for part in participants:
            sc = LLM.tokenizer.encode(part)
            sc = torch.cat((newline_token, sc), dim=1)
            stop_conditions.append((sc, "\n" + part))
            stop_conditions.append((sc, "\n " + part))

    prompt = prompt.strip()

    if len(prompt) == 0:
        # No prompt given
        generator.gen_begin_empty()
        prompt_tok = 0
    else:
        # Set the context/user input
        prompt_tokens = LLM.tokenizer.encode(prompt)
        generator.gen_begin_reuse(prompt_tokens)
        prompt_tok = prompt_tokens.shape[-1]

    compl_toks = [0] # list needed to pass by ref.

    output = ""
    # if stream:
    #     yield prompt + " "
    #     yield from _stream_helper(generator, stop_conditions, max_tokens, compl_toks)
    #     t1.done("Stream complete.")

    # Combine all yields.
    output = "".join(list(_stream_helper(generator, stop_conditions, max_tokens, compl_toks)))
    t1.done("Response: {}", output)

    return (
        LLM.modelName,
        output,
        prompt_tok,
        compl_toks[0],
        round(1000 * t1.elapsed())
    )


def _stream_helper(generator, stop_conditions, max_tokens, total_tokens):
    """ Generate model response using beam search. """

    # Generate loop
    # Beam search uses conditional probability to find the
    # best output tokens
    generator.begin_beam_search()

    res_line = ""
    chunk_size = 2
    stop_condition = False
    held_text = ""
    num_res_tokens = 0

    for i in range(max_tokens):
        # Truncate the past if the next chunk might generate past max_seq_length
        if generator.sequence_actual is not None:
            nextgen = generator.sequence_actual.shape[-1] + chunk_size + \
                generator.settings.beam_length + 1
            if nextgen > LLM.model.config.max_seq_len:
                generator.gen_prune_left(chunk_size)

        # Get the most probable token and append to sequence
        gen_token = generator.beam_search()

        # If token is EOS, replace it with newline before continuing
        if gen_token.item() == LLM.tokenizer.eos_token_id:
            generator.replace_last_token(LLM.tokenizer.newline_token_id)

        # Decode current line to get new characters added
        # (decoding a single token gives incorrect results sometimes
        # due to how SentencePiece works)
        prev_res_line = res_line
        num_res_tokens += 1
        res_line = LLM.tokenizer.decode(generator.sequence_actual[0, -num_res_tokens:])
        new_text = res_line[len(prev_res_line):]

        # Since SentencePiece is slightly ambiguous,
        # the first token produced after a newline may not be the
        # same that is reproduced when we encode the text later,
        # even though it encodes the same string
        if num_res_tokens == 1 and len(new_text) > 0:
            replace = LLM.tokenizer.encode(new_text)[0]
            if replace.shape[-1] == 1:
                generator.replace_last_token(replace)

        # Delay streaming if new text might be part of a stop condition
        hold_text = False
        for _, stop_string in stop_conditions:
            if stop_string.lower().startswith((held_text + new_text).lower()):
                hold_text = True

        # Stream to client
        if not hold_text:
            packet = held_text + new_text
            yield packet
            held_text = ""
        else:
            held_text += new_text

        # Check the stop conditions
        if gen_token.item() == LLM.tokenizer.eos_token_id:
            if len(held_text) > 0:  # Not sure if this could actually happen
                plen = LLM.tokenizer.encode(held_text).shape[-1]
                res_line = res_line[:-len(held_text)]
                generator.gen_rewind(plen)
            stop_condition = True
            break

        for stop_tokens, stop_string in stop_conditions:
            if res_line.lower().endswith(stop_string.lower()):
                generator.gen_rewind(
                    stop_tokens.shape[-1] - (
                        1 if stop_tokens[0, 0].item() == LLM.tokenizer.newline_token_id else 0
                    )
                )
                res_line = res_line[:-len(stop_string)]
                stop_condition = True
                break

        if stop_condition:
            break

    generator.end_beam_search()
    res_line = res_line.strip()
    total_tokens[0] += num_res_tokens
    return res_line


if __name__ == "__main__":
    model = "/home/user/models/TheBloke_stable-vicuna-13B-GPTQ/stable-vicuna-13B-GPTQ-4bit.compat.no-act-order.safetensors"
    init_exllama_model(model)
    LLM.user = "### Human:"
    LLM.bot = "### Assistant:"

    # print(simple("### Human: Hello there!\n### Assistant:"))

    prompt = """
    A chat between two friends where one of them is a bit too eager to share their opinions and experiences.
    ### Human: What is the meaning of life?
    ### Assistant:"""

    # print("Streaming ...")
    # for resp in get_exllama_response(prompt, stream=True):
    #     print(resp, end="")
    #     sys.stdout.flush()

    print("Printing ...")
    print(get_exllama_response(prompt, stream=False))
