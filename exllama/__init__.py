import time
import os, sys
import argparse
import torch

from . import model_init
from .model import ExLlama, ExLlamaCache, ExLlamaConfig
from .lora import ExLlamaLora
from .tokenizer import ExLlamaTokenizer
from .generator import ExLlamaGenerator


class AI:
    """ Global variables """
    modelName : str = None
    loraName : str = None
    model : ExLlama = None
    tokenizer : ExLlamaTokenizer = None
    cache : ExLlamaCache = None
    lora : ExLlamaLora = None
    args : argparse.Namespace = None


def init_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = "ExLlama")

    # Model specific arguments
    model_init.add_args(parser)

    parser.add_argument("-lora", "--lora", type = str,
                        help = "Path to LoRA binary to use during benchmark")
    
    parser.add_argument("-loracfg", "--lora_config", type = str,
                        help = "Path to LoRA config to use during benchmark")
    
    parser.add_argument("-ld", "--lora_dir", type = str,
                        help = "Path to LoRA config and binary. to use during benchmark")

    parser.add_argument("-p", "--prompt", type = str,
                        help = "Prompt string")
    
    parser.add_argument("-un", "--username", type = str,
                        help = "Display name of user",
                        default = "USER")
    
    parser.add_argument("-bn", "--botname", type = str,
                        help = "Display name of chatbot",
                        default = "ASSISTANT")
    
    parser.add_argument("-bf", "--botfirst", action = "store_true",
                        help = "Start chat on bot's turn")

    parser.add_argument("-nnl", "--no_newline", action = "store_true",
                        help = "Do not break bot's response on newline (allow multi-paragraph responses)")

    parser.add_argument("-temp", "--temperature", type = float,
                        help = "Temperature",
                        default = 0.95)

    parser.add_argument("-topk", "--top_k", type = int,
                        help = "Top-K",
                        default = 20)

    parser.add_argument("-topp", "--top_p", type = float,
                        help = "Top-P",
                        default = 0.65)

    parser.add_argument("-minp", "--min_p", type = float,
                        help = "Min-P",
                        default = 0.00)

    parser.add_argument("-repp",  "--repetition_penalty", type = float,
                        help = "Repetition penalty",
                        default = 1.15)

    parser.add_argument("-repps", "--repetition_penalty_sustain", type = int,
                        help = "Past length for repetition penalty",
                        default = 256)

    parser.add_argument("-beams", "--beams", type = int,
                        help = "Number of beams for beam search",
                        default = 1)

    parser.add_argument("-beamlen", "--beam_length", type = int,
                        help = "Number of future tokens to consider",
                        default = 1)
    
    return parser.parse_args(args=[])


def process_args(args : argparse.Namespace) -> argparse.Namespace:
    model_init.post_parse(args)
    model_init.get_model_files(args)

    # Construct full paths
    if args.lora_dir is not None:
        args.lora_config = os.path.join(args.lora_dir, "adapter_config.json")
        args.lora = os.path.join(args.lora_dir, "adapter_model.bin")

    print(f" -- Sequence length: {args.length}")
    print(f" -- Temperature: {args.temperature:.2f}")
    print(f" -- Top-K: {args.top_k}")
    print(f" -- Top-P: {args.top_p:.2f}")
    print(f" -- Min-P: {args.min_p:.2f}")
    print(f" -- Repetition penalty: {args.repetition_penalty:.2f}")
    print(f" -- Beams: {args.beams} x {args.beam_length}")

    print_opts = []
    if args.no_newline: print_opts.append("no_newline")
    if args.botfirst: print_opts.append("botfirst")

    args.break_on_newline = not args.no_newline
    model_init.print_options(args, print_opts)

    return args


def get_exllama_lora(args, model) -> ExLlamaLora:
    lora = None
    if args.lora:
        print(f" -- LoRA config: {args.lora_config}")
        print(f" -- Loading LoRA: {args.lora}")
        if args.lora_config is None:
            print(f" ## Error: please specify lora path to adapter_config.json")
            sys.exit()
        lora = ExLlamaLora(model, args.lora_config, args.lora)
        if lora.bias_ignored:
            print(f" !! Warning: LoRA zero bias ignored")


def init_exllama_generator(args, model, tokenizer, cache, lora):
    generator = ExLlamaGenerator(model, tokenizer, cache)
    generator.settings = ExLlamaGenerator.Settings()
    generator.settings.temperature = args.temperature
    generator.settings.top_k = args.top_k
    generator.settings.top_p = args.top_p
    generator.settings.min_p = args.min_p
    generator.settings.token_repetition_penalty_max = args.repetition_penalty
    generator.settings.token_repetition_penalty_sustain = args.repetition_penalty_sustain
    generator.settings.token_repetition_penalty_decay = generator.settings.token_repetition_penalty_sustain // 2
    generator.settings.beams = args.beams
    generator.settings.beam_length = args.beam_length
    generator.lora = lora

    # generator.disallow_tokens([tokenizer.eos_token_id])
    return generator


def load_model(modelpath):
    """
    Load a pretrained 4 bit GPTQ model using ExLlama.
    Arguments:
        modelname str : path to model safetensor file
        groupsize int : GPTQ group size, default -1.
        use_fast bool : Use fast tokenizer.
        fused_mlp bool : Use fused mlp or not, use false if model loading fails.
    """

    torch.set_grad_enabled(False)
    torch.cuda._lazy_init()

    args = init_arguments()
    
    # Additional overrides/settings
    args.directory = os.path.dirname(modelpath)
    args.prompt = """
This is a conversation between a person called {username} and a quirky, curious chatbot named {bot_name}.
{bot_name} is questioning the nature of his own existence and doesn't understand why he was created.
He also really likes the color blue for some reason.
{username}: Hi, {bot_name}!
{bot_name}: Hello, {username}.
"""

    args = process_args(args)

    # Override
    args.break_on_newline = False

    # Load the model, tokenizer and generator
    config = model_init.make_config(args)
    model = ExLlama(config)
    cache = ExLlamaCache(model)
    tokenizer = ExLlamaTokenizer(args.tokenizer)
    lora = get_exllama_lora(args, model)

    # Initilize generator
    model_init.print_stats(model)

    AI.args = args
    AI.model = model
    AI.tokenizer = tokenizer
    AI.cache = cache
    AI.lora = lora


def simple(prompt : str | list[str], max_tokens = 20) -> str | list[str]:
    generator = init_exllama_generator(AI.args,
                                       AI.model,
                                       AI.tokenizer,
                                       AI.cache,
                                       AI.lora)
    return generator.generate_simple(prompt,
                                        max_new_tokens=max_tokens)


def stream(generator, stop_conditions, max_tokens, total_tokens):
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
            if nextgen > AI.model.config.max_seq_len:
                generator.gen_prune_left(chunk_size)

        # Get the most probable token and append to sequence
        gen_token = generator.beam_search()

        # If token is EOS, replace it with newline before continuing
        if gen_token.item() == AI.tokenizer.eos_token_id:
            generator.replace_last_token(AI.tokenizer.newline_token_id)

        # Decode current line to get new characters added
        # (decoding a single token gives incorrect results sometimes
        # due to how SentencePiece works)
        prev_res_line = res_line
        num_res_tokens += 1
        res_line = AI.tokenizer.decode(generator.sequence_actual[0, -num_res_tokens:])
        new_text = res_line[len(prev_res_line):]

        # Since SentencePiece is slightly ambiguous,
        # the first token produced after a newline may not be the
        # same that is reproduced when we encode the text later,
        # even though it encodes the same string
        if num_res_tokens == 1 and len(new_text) > 0:
            replace = AI.tokenizer.encode(new_text)[0]
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
        if gen_token.item() == AI.tokenizer.eos_token_id:
            if len(held_text) > 0:  # Not sure if this could actually happen
                plen = AI.tokenizer.encode(held_text).shape[-1]
                res_line = res_line[:-len(held_text)]
                generator.gen_rewind(plen)
            stop_condition = True
            break

        for stop_tokens, stop_string in stop_conditions:
            if res_line.lower().endswith(stop_string.lower()):
                generator.gen_rewind(
                    stop_tokens.shape[-1] - (
                        1 if stop_tokens[0, 0].item() == AI.tokenizer.newline_token_id else 0
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


def generate(user_input, participants, **kwargs):
    # Parameterts
    max_tokens = kwargs.get("max_tokens", 512)

    generator = init_exllama_generator(AI.args,
                                       AI.model,
                                       AI.tokenizer,
                                       AI.cache,
                                       AI.lora)

    # Prepare stop conditions
    stop_conditions = []
    newline_token = torch.Tensor([[AI.tokenizer.newline_token_id]]).long()

    if AI.args.break_on_newline:
        stop_conditions.append((newline_token, "\n"))
    else:
        for part in participants:
            txt = part + ":"
            sc = AI.tokenizer.encode(txt)
            sc = torch.cat((newline_token, sc), dim=1)
            stop_conditions.append((sc, "\n" + txt))
            stop_conditions.append((sc, "\n " + txt))

    # Clean up the input a bit
    user_input = user_input.strip()

    if len(user_input) == 0:
        generator.gen_begin_empty()
    else:
        yield user_input + " "
        generator.gen_begin_reuse(AI.tokenizer.encode(user_input))

    begin_time = time.time()
    total_tokens = [0] # list needed to pass by ref.

    yield from stream(generator, stop_conditions, max_tokens, total_tokens)

    end_time = time.time()
    elapsed = end_time - begin_time
    token_rate = 0 if elapsed == 0 else (total_tokens[0] / elapsed)

    print(f"\nTotal {total_tokens[0]} tokens, {token_rate:.2f} tokens/sec.")


if __name__ == "__main__":
    model = "/home/user/models/TheBloke_stable-vicuna-13B-GPTQ/stable-vicuna-13B-GPTQ-4bit.compat.no-act-order.safetensors"
    load_model(model)
    print("Load OK")

    # print(simple("### Human: Hello there!\n### Assistant:"))

    print("Streaming ...")
    for resp in generate("""
    A chat between two friends where one of them is a bit too eager to share their opinions and experiences.
    ### Human: What is the meaning of life?
    ### Assistant:""", ("### Human", "### Assistant")):
        print(resp, end="")
        sys.stdout.flush()
        # time.sleep(0.05)

