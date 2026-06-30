import torch.nn as nn
import yaml
import torch
import json


""" configuration json """
class Config(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    @classmethod
    def load(cls, file):
        with open(file, 'r') as f:
            config = json.loads(f.read())
            return Config(config)
        

def process_activation(config):
    """Replaces activation name string with activation function class in config."""
    if "act" in config:
        act_name = config["act"].lower()
        if act_name == "gelu":
            config["act"] = nn.GELU
        elif act_name == "relu":
            config["act"] = nn.ReLU
        else:
            raise ValueError(f"Activation {act_name} not supported.")


def model_config(config_path=None):
    if config_path is not None:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        process_activation(config)
        return Config(**config)
    else:
        raise ValueError("config path is None")

    
def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')