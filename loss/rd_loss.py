import torch
import torch.nn as nn


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        
        out = self.loss_fn(outputs, targets)
        
        return out