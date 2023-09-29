""" Yaml based configuration management.
    Settings are grouped by sections defined below.
"""
import yaml
from dataclasses import dataclass


@dataclass
class TextGenConfig:
    user_fmt : str = "### Human:"
    bot_fmt : str = "### Assistant:"
    instruction_fmt : str = ""
    context_length : int = 4096

@dataclass
class ModelConfig:
    model_file_path : str = "models/<folder>/model.st"
    lora_file_path : str = "loras/<folder>/lora.st"

@dataclass
class APIConfig:
    polyai_api_key : str = "pl-test"
    polyai_api_base : str = "http://localhost:8001/polyai/"

@dataclass
class ServerConfig:
    api_endpoint_port : int = 8001
    ssl_endpoint_port : int = 8002

@dataclass
class PostgresConfig:
    # Postgres configurations to store api_keys and requests.
    db_user : str = ""
    db_pass : str = ""
    db_host : str = ""
    db_port : int = 5432
    db_name : str = "polyai"

@dataclass
class DockerConfig:
    # Docker specific variables if server is run in a docker container.
    server_cache : str = "~/.cache"
    models_dir : str = "models/"
    docker_network : str = "polyai"


# List of sections in the YAML file.
_sections : list = []

def _api_settings():
    _sections.append(APIConfig())

def _server_settings():
    _sections.append(TextGenConfig())
    _sections.append(ModelConfig())
    _sections.append(APIConfig())
    _sections.append(ServerConfig())
    _sections.append(PostgresConfig())
    _sections.append(DockerConfig())


def _load_settings(settings_yaml: str = 'settings.yaml') -> bool:
    """ Load settings from a yaml file. Returns True if load was successful. """
    _yaml = {}
    try:
        with open(settings_yaml) as fp:
            _yaml = yaml.safe_load(fp)
            print("Load OK:", settings_yaml)
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
        print("Save OK:", settings_yaml)


def load_api_settings(settings_yaml: str = 'settings.yaml') -> bool:
    _api_settings()
    return _load_settings(settings_yaml)

def save_api_settings(settings_yaml: str = 'settings.yaml') -> bool:
    _api_settings()
    return _save_settings(settings_yaml)

def load_server_settings(settings_yaml: str = 'settings.yaml') -> bool:
    _server_settings()
    return _load_settings(settings_yaml)

def save_server_settings(settings_yaml: str = 'settings.yaml') -> bool:
    _server_settings()
    return _save_settings(settings_yaml)

