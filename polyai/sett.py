""" Yaml based configuration management.
    Settings are grouped by sections defined below.
"""
import yaml
from dataclasses import dataclass


@dataclass
class text_generation:
    user_fmt : str = "### Human:"
    bot_fmt : str = "### Assistant:"
    instruction_fmt : str = ""
    context_length : int = 4096

TextGen = text_generation()

@dataclass
class models:
    model_file_path : str = None
    lora_file_path : str = None
    bert_file_path : str = None
    vram_config : str = "8,10,10,10"
    bert_device : str = "cuda"

Model = models()

@dataclass
class api_config:
    polyai_api_key : str = "pl-test"
    polyai_api_base : str = "http://localhost:8001/polyai/"
    ssh_tunnel_host : str = None
    ssh_tunnel_user : str = None
    ssh_tunnel_port : int = 22
    ssh_tunnel_pass : str = None

API = api_config()

@dataclass
class server:
    api_endpoint_port : int = 8001
    stream_endpoint_port : int = 8002
    use_ssl : bool = False
    listen_all : bool = False       # listen to 0.0.0.0
    log_level : int = 8
    debug : bool = False
    log_file_name : str = "polyai.log"
    log_append : bool = False

Server = server()


@dataclass
class postgres_db:
    # Postgres configurations to store api_keys and requests.
    db_user : str = ""
    db_pass : str = ""
    db_host : str = ""
    db_port : int = 5432
    db_name : str = "polyai"

Postgres = postgres_db()


@dataclass
class docker_config:
    # Docker specific variables if server is run in a docker container.
    server_cache : str = "~/.cache"
    models_dir : str = "models/"
    docker_network : str = "polyai"

Docker = docker_config()

# List of sections in the YAML file.
_sections : list = []

def _api_settings():
    _sections.append(API)

def _server_settings():
    _sections.append(TextGen)
    _sections.append(Model)
    _sections.append(Server)
    _sections.append(Postgres)
    _sections.append(Docker)


def _load_settings(settings_yaml: str = 'settings.yaml') -> bool:
    """ Load settings from a yaml file. Returns True if load was successful. """
    _yaml = {}
    try:
        with open(settings_yaml) as fp:
            _yaml = yaml.safe_load(fp)
            # print("Load OK:", settings_yaml)
    except: return False
    
    for section in _sections:
        section.__dict__.update(
            _yaml.get(section.__class__.__name__, section.__dict__))

    return True


def _save_settings(settings_yaml: str = 'settings.yaml'):
    """ Save current settings to a yaml file. """
    d = {
        section.__class__.__name__ : section.__dict__
        for section in _sections
    }

    with open(settings_yaml, 'w') as fp:
        yaml.safe_dump(d, fp, sort_keys=False, indent=4)
        # print("Save OK:", settings_yaml)


def load_api_settings(settings_yaml: str = 'settings.yaml') -> bool:
    _api_settings()
    return _load_settings(settings_yaml)

def save_api_settings(settings_yaml: str = 'settings.yaml'):
    _api_settings()
    return _save_settings(settings_yaml)

def load_server_settings(settings_yaml: str = 'settings.yaml') -> bool:
    _server_settings()
    return _load_settings(settings_yaml)

def save_server_settings(settings_yaml: str = 'settings.yaml'):
    _server_settings()
    return _save_settings(settings_yaml)

