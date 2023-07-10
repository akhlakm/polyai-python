""" Global server states. """

import torch

class Server:
    @classmethod
    def vram_usage(cls):
        free = 0
        total = 0
        devices = torch.cuda.device_count()
        for i in range(devices):
            mem = torch.cuda.mem_get_info(i)
            free += mem[0]/1024/1024/1024
            total += mem[1]/1024/1024/1024
        return devices, total - free, total, free


class LLM:
    _loader = None
    _model_name : str = None
    _lora_name : str = None
    _model = None
    _tokenizer = None
    _cache = None
    _lora = None
    _system_name : str = ""
    _user_name : str = None
    _bot_name : str = None
    _is_ready : bool = False
    _stop_generation : bool = False
    _break_on_newline = False

    @classmethod
    def stop_generation(cls):
        cls._stop_generation = True

    @classmethod
    def model_name(cls):
        return cls._model_name

    @classmethod
    def unload_model(cls):
        if not cls._is_ready and cls._model is not None:
            cls.stop_generation()
        cls._is_ready = False
        cls._model = None
        cls._tokenizer = None
        cls._lora = None
    
    @classmethod
    def get_available_models(cls):
        return []
    
    @classmethod
    def load_model(cls, model_file):
        cls._loader.load_model(model_file)

    @classmethod
    def use_lora(cls, lora_dir):
        cls._loader.add_lora(lora_dir)

    @classmethod
    def generate(cls, prompt, params = {}):
        if not cls._is_ready:
            raise ConnectionError
        return cls._loader.generate(prompt, params)

    @classmethod
    def stream(cls, prompt, params = {}):
        if not cls._is_ready:
            raise ConnectionError
        yield from cls._loader.stream(prompt, params)

    @classmethod
    def encode(cls, string, **kwargs):
        return cls._tokenizer.encode(string)

    @classmethod
    def decode(cls, string, **kwargs):
        return cls._tokenizer.decode(string)[0]

    @classmethod
    def parameters(body : dict, chat=False):
        if 'max_tokens' in body:
            body['max_new_tokens'] = body['max_tokens']

        generate_params = {
            'max_new_tokens': int(body.get('max_new_tokens', body.get('max_length', 200))),
            'do_sample': bool(body.get('do_sample', True)),
            'temperature': float(body.get('temperature', 0.5)),
            'top_p': float(body.get('top_p', 1)),
            'min_p': float(body.get('min_p', 0.0)),
            'typical_p': float(body.get('typical_p', body.get('typical', 1))),
            'epsilon_cutoff': float(body.get('epsilon_cutoff', 0)),
            'eta_cutoff': float(body.get('eta_cutoff', 0)),
            'tfs': float(body.get('tfs', 1)),
            'top_a': float(body.get('top_a', 0)),
            'repetition_penalty': float(body.get('repetition_penalty', body.get('rep_pen', 1.1))),
            'repetition_penalty_range': int(body.get('repetition_penalty_range', 0)),
            'encoder_repetition_penalty': float(body.get('encoder_repetition_penalty', 1.0)),
            'top_k': int(body.get('top_k', 0)),
            'min_length': int(body.get('min_length', 0)),
            'no_repeat_ngram_size': int(body.get('no_repeat_ngram_size', 0)),
            'num_beams': int(body.get('num_beams', 1)),
            'penalty_alpha': float(body.get('penalty_alpha', 0)),
            'length_penalty': float(body.get('length_penalty', 1)),
            'early_stopping': bool(body.get('early_stopping', False)),
            'mirostat_mode': int(body.get('mirostat_mode', 0)),
            'mirostat_tau': float(body.get('mirostat_tau', 5)),
            'mirostat_eta': float(body.get('mirostat_eta', 0.1)),
            'seed': int(body.get('seed', -1)),
            'add_bos_token': bool(body.get('add_bos_token', True)),
            'truncation_length': int(body.get('truncation_length', body.get('max_context_length', 2048))),
            'ban_eos_token': bool(body.get('ban_eos_token', False)),
            'skip_special_tokens': bool(body.get('skip_special_tokens', True)),
            'custom_stopping_strings': '',  # leave this blank
            'stopping_strings': body.get('stopping_strings', []),
        }

        preset_name = body.get('preset', 'None')
        if preset_name not in ['None', None, '']:
            raise NotImplementedError("preset not implemented in polyai")

        if chat:
            raise NotImplementedError("chat not implemented in polyai")

        return generate_params
