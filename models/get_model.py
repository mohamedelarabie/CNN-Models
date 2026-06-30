import torch
from models.LeNet1 import model as LeNet1
from models.LeNet4 import model as LeNet4
from loss.rd_loss import CrossEntropyLoss

def get_model(config):
    model_name = config['name']
    if model_name == 'LeNet1':
        net = LeNet1(**config)
        loss = CrossEntropyLoss()
    elif model_name == 'LeNet4':
        net = LeNet4(**config)
        loss = CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    

    return net, loss