import yaml
from typing import (
    List,
    Dict,
)
from dataclasses import dataclass

@dataclass
class ExperimentConfig:
    model_name: str
    experiment_name: str
    description: str
    num_tasks: int
    num_runners: int
    generate_prompts: bool = False
    num_prompts: int = None
    prompt_length: int = None
    max_tokens: int = None
    prompts_file: str = None
    seed: int = 42
    node_url: str = None
    account_address: str = None

def load_yaml_configs(file_path: str) -> List[ExperimentConfig]:
    with open(file_path, 'r') as file:
        config_data = yaml.safe_load(file)

    if not isinstance(config_data, list):
        config_data = [config_data]

    return [ExperimentConfig(**experiment) for experiment in config_data]
