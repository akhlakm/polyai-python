import os, sys
import argparse
import torch

import pylogg
import polyai.server.state as state
from polyai.server.exllama import model_init
from polyai.server.exllama.model import ExLlama, ExLlamaCache
from polyai.server.exllama.lora import ExLlamaLora
from polyai.server.exllama.tokenizer import ExLlamaTokenizer
from polyai.server.exllama.generator import ExLlamaGenerator

EPS = 1e-10
log = pylogg.New("llm")

class ExllamaModel:
    def __init__(self, vram_spec = None) -> None:
        self.vram_spec = vram_spec

        state.LLM._user_name = os.getenv("POLYAI_USER_FMT", "USER:")
        state.LLM._bot_name = os.getenv("POLYAI_BOT_FMT", "ASSISTANT:")

        # Not sure what this does exactly.
        torch.set_grad_enabled(False)
        torch.cuda._lazy_init()
        # if torch.version.hip:
        #     config.rmsnorm_no_half2 = True
        #     config.rope_no_half2 = True
        #     config.matmul_no_half2 = True
        #     config.silu_no_half2 = True


    def print_vram_usage(self):
        log.info(
            "VRAM usage, %d GPUs: %0.4f GB / %0.4f GB (%0.4f GB free)"
            %state.Server.vram_usage())


    def load_model(self, model_file):
        # Notify user
        t1 = log.trace("Loading ExLlama model: {}", model_file)
        self.print_vram_usage()

        # Default exllama options
        parser = argparse.ArgumentParser(description = "ExLlama")
        model_init.add_args(parser)
        exargs = parser.parse_args(args=[])

        # Overrides/settings
        exargs.directory = os.path.dirname(model_file)
        if self.vram_spec is not None:
            log.info("Using GPU map: {} GB", self.vram_spec)
            exargs.gpu_split = self.vram_spec

        # Unload existing model if any
        state.LLM.unload_model()

        # Post process the arguments
        model_init.get_model_files(exargs)

        # Load the model, tokenizer
        config = model_init.make_config(exargs)
        state.LLM._model = ExLlama(config)
        state.LLM._cache = ExLlamaCache(state.LLM._model)
        state.LLM._tokenizer = ExLlamaTokenizer(exargs.tokenizer)

        state.LLM._model_name = os.path.basename(model_file).split(".")[0]
        state.LLM._is_ready = True

        model_init.print_stats(state.LLM._model)
        t1.done("Model loaded: {}", state.LLM._model_name)
        self.print_vram_usage()


    def add_lora(self, lora_dir):
        assert state.LLM._model is not None, "Model must be loaded first"

        t1 = log.info("Loading LoRA from: {}", lora_dir)
        lora_config = os.path.join(lora_dir, "adapter_config.json")
        lora_bin = os.path.join(lora_dir, "adapter_model.bin")

        lora = ExLlamaLora(state.LLM._model, lora_config, lora_bin)
        state.LLM._lora = lora
        state.LLM._lora_name = os.path.basename(lora_dir)

        t1.done("Lora loaded: {}", lora_dir)
        if lora.bias_ignored:
            log.warn("LoRA zero bias ignored")


    def generate(self, prompt, params):
        """
        Given a prompt message, generate model response.
        
        Returns:
            Name of the model,
            List of generated responses,
            Total input tokens,
            Total completion tokens,
            Total time elapsed in miliseconds.
        """
        t1 = log.trace("Getting LLM response.")
        generator, stops, max_tokens = _prepare_generation(params)
        prompt = prompt.strip()
        state.LLM._is_ready = False

        # Set the context/user input
        prompt_tokens = state.LLM.encode(prompt)
        generator.gen_begin_reuse(prompt_tokens)
        prompt_tok = prompt_tokens.shape[-1]
        compl_toks = [0] # list needed to pass by ref.

        print("\n", "-"*80)
        print(prompt)

        output = ""
        for out in _stream_helper(generator, stops, max_tokens, compl_toks):
            output += out
            print(out, end="", flush=True)

        state.LLM._is_ready = True

        print("\n", "-"*80, "\n")
        t1.done("Generation done.")
        log.trace("Response message: {}", output)

        return (
            state.LLM.model_name(),
            [output],
            prompt_tok,
            compl_toks[0],
            round(1000 * t1.elapsed())
        )


    def stream(self, prompt, params):
        """
        Given a prompt message, stream model response.

        """
        t1 = log.trace("Streaming LLM response.")
        generator, stops, max_tokens = _prepare_generation(params)
        prompt = prompt.strip()
        state.LLM._is_ready = False

        # Set the context/user input
        prompt_tokens = state.LLM.encode(prompt)
        generator.gen_begin_reuse(prompt_tokens)
        prompt_tok = prompt_tokens.shape[-1]
        compl_toks = [0] # list needed to pass by ref.

        yield prompt + " "
        yield from _stream_helper(generator, stops, max_tokens, compl_toks)
        state.LLM._is_ready = True
        t1.done("Stream complete.")


def _prepare_generation(param):
    print(param)
    param = state.LLM.parameters(param)
    generator = ExLlamaGenerator(state.LLM._model,
                                    state.LLM._tokenizer, state.LLM._cache)
    generator.settings = ExLlamaGenerator.Settings()
    generator.settings.temperature = param['temperature']
    generator.settings.top_k = param['top_k']
    generator.settings.top_p = param['top_p']
    generator.settings.min_p = param['min_p']
    generator.settings.beams = param['num_beams']
    generator.settings.token_repetition_penalty_max = param['repetition_penalty']

    if param['repetition_penalty_range'] <= 0:
        generator.settings.token_repetition_penalty_sustain = -1
    else:
        generator.settings.token_repetition_penalty_sustain = param['repetition_penalty_range']

    if param['ban_eos_token']:
        generator.disallow_tokens([state.LLM._tokenizer.eos_token_id])
    else:
        generator.disallow_tokens(None)

    generator.lora = state.LLM._lora
    log.trace("Generation settings: {}", str(generator.settings.__dict__))

    max_tokens = param['max_new_tokens']
    log.trace("Max tokens: {}", max_tokens)

    participants = [state.LLM._user_name, state.LLM._bot_name]
    if len(state.LLM._system_name) > 0:
        participants.append(state.LLM._system_name)
    log.trace("Participants: {}", participants)

    # Prepare stop conditions
    stop_conditions = []
    newline_token = torch.Tensor([[state.LLM._tokenizer.newline_token_id]]).long()

    if state.LLM._break_on_newline:
        # Stop generation on newline character.
        stop_conditions.append((newline_token, "\n"))
    else:
        # Stop generation if a newline followed by a participant is generated.
        for part in participants:
            sc = state.LLM.encode(part)
            sc = torch.cat((newline_token, sc), dim=1)
            stop_conditions.append((sc, "\n" + part))
            stop_conditions.append((sc, "\n " + part))
        # Other stopping strings requested
        for pattern in param['stopping_strings']:
            sc = state.LLM.encode(pattern)
            stop_conditions.append((sc, pattern))

    return generator, stop_conditions, max_tokens


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
            if nextgen > state.LLM._model.config.max_seq_len:
                generator.gen_prune_left(chunk_size)

        # Get the most probable token and append to sequence
        gen_token = generator.beam_search()

        # If token is EOS, replace it with newline before continuing
        if gen_token.item() == state.LLM._tokenizer.eos_token_id:
            generator.replace_last_token(state.LLM._tokenizer.newline_token_id)

        # Decode current line to get new characters added
        # (decoding a single token gives incorrect results sometimes
        # due to how SentencePiece works)
        prev_res_line = res_line
        num_res_tokens += 1
        res_line = state.LLM.decode(generator.sequence_actual[0, -num_res_tokens:])
        new_text = res_line[len(prev_res_line):]

        # Since SentencePiece is slightly ambiguous,
        # the first token produced after a newline may not be the
        # same that is reproduced when we encode the text later,
        # even though it encodes the same string
        if num_res_tokens == 1 and len(new_text) > 0:
            replace = state.LLM.encode(new_text)[0]
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
        if gen_token.item() == state.LLM._tokenizer.eos_token_id:
            if len(held_text) > 0:  # Not sure if this could actually happen
                plen = state.LLM.encode(held_text).shape[-1]
                res_line = res_line[:-len(held_text)]
                generator.gen_rewind(plen)
            stop_condition = True
            break

        for stop_tokens, stop_string in stop_conditions:
            if res_line.lower().endswith(stop_string.lower()):
                generator.gen_rewind(
                    stop_tokens.shape[-1] - (
                        1 if stop_tokens[0, 0].item() == state.LLM._tokenizer.newline_token_id else 0
                    )
                )
                res_line = res_line[:-len(stop_string)]
                stop_condition = True
                break

        if stop_condition or state.LLM._stop_generation:
            break

    generator.end_beam_search()
    res_line = res_line.strip()
    total_tokens[0] += num_res_tokens
    return res_line


def init_exllama(args):
    if args.vram:
        assert "," in args.vram, "--vram must be a comma separated string"
        assert " " not in args.vram, "--vram must be without any space"

    state.LLM._loader = ExllamaModel(args.vram)
    return state.LLM._loader
