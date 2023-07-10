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
    def get(cls, name, ptype, default, d : dict):
        """ Update a dictionary item by typecasting. """
        value = d.get(name, None)
        if value is not None:
            try:
                value = ptype(value)
            except:
                raise TypeError("invalid type for {name}")
        else:
            value = default
        return value

    @classmethod
    def parameters(cls, body : dict, chat=False):
        # Aliases
        if 'max_tokens' in body:
            body['max_new_tokens'] = cls.get('max_tokens', int, None, body)
        elif 'max_length' in body:
            body['max_new_tokens'] = cls.get('max_length', int, None, body)
        if 'typical' in body:
            body['typical_p'] = cls.get('typical', float, None, body)
        if 'rep_pen' in body:
            body['repetition_penalty'] = cls.get('rep_pen', float, None, body)
        if 'max_context_length' in body:
            body['truncation_length'] = cls.get('max_context_length', int, None, body)

        generate_params = {
            'max_new_tokens':               cls.get('max_new_tokens', int, 512, body),
            'do_sample':                    cls.get('do_sample', bool, True, body),
            'temperature':                  cls.get('temperature', float, 0.5, body),
            'top_p':                        cls.get('top_p', float, 0.95, body),
            'min_p':                        cls.get('min_p', float, 0.00, body),
            'typical_p':                    cls.get('typical_p', float, 1, body),
            'epsilon_cutoff':               cls.get('epsilon_cutoff', float, 0, body),
            'eta_cutoff':                   cls.get('eta_cutoff', float, 0, body),
            'tfs':                          cls.get('tfs', float, 1, body),
            'top_a':                        cls.get('top_a', float, 0, body),
            'repetition_penalty':           cls.get('repetition_penalty', float, 1.1, body),
            'repetition_penalty_range':     cls.get('repetition_penalty_range', int, 0, body),
            'encoder_repetition_penalty':   cls.get('encoder_repetition_penalty', float, 1.0, body),
            'top_k':                        cls.get('top_k', int, 0, body),
            'min_length':                   cls.get('min_length', int, 0, body),
            'no_repeat_ngram_size':         cls.get('no_repeat_ngram_size', int, 0, body),
            'num_beams':                    cls.get('num_beams', int, 1, body),
            'penalty_alpha':                cls.get('penalty_alpha', float, 0, body),
            'length_penalty':               cls.get('length_penalty', float, 1, body),
            'early_stopping':               cls.get('early_stopping', bool, False, body),
            'mirostat_mode':                cls.get('mirostat_mode', int, 0, body),
            'mirostat_tau':                 cls.get('mirostat_tau', float, 5, body),
            'mirostat_eta':                 cls.get('mirostat_eta', float, 0.1, body),
            'seed':                         cls.get('seed', int, -1, body),
            'add_bos_token':                cls.get('add_bos_token', bool, True, body),
            'truncation_length':            cls.get('truncation_length', int, 2048, body),
            'ban_eos_token':                cls.get('ban_eos_token', bool, False, body),
            'skip_special_tokens':          cls.get('skip_special_tokens', bool, True, body),
            'custom_stopping_strings': '',  # leave this blank
            'stopping_strings':             cls.get('stopping_strings', list, [], body),
        }

        preset_name = body.get('preset', 'None')
        if preset_name not in ['None', None, '']:
            raise NotImplementedError("preset not implemented in polyai")

        if chat:
            raise NotImplementedError("chat not implemented in polyai")

        return generate_params


class BERT:
    _loader = None
    _model_name : str = None
    _pipeline = None
    _is_ready : bool = False
    _stop_generation : bool = False

    @classmethod
    def stop_generation(cls):
        cls._stop_generation = True

    @classmethod
    def model_name(cls):
        return cls._model_name

    @classmethod
    def ner_tags(cls, text):
        """
        Perform NER using the loaded BERT model on the given text.
        Returns:
            Model name,
            List of generated NER tags as a dict format,
            Total time elapsed in miliseconds.
        """
        return cls._loader.ner_tags(text)
