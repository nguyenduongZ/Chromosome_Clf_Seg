import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedFocalLoss(nn.Module):
    def __init__(self, gammas=[0, 1], alpha=0.25, reduction='mean'):
        super().__init__()
        self.gammas = gammas
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, _input, target, weight = None):
        pt = torch.sigmoid(_input)
        
        loss = torch.zeros_like(_input)
        for gamma in self.gammas:
            loss += - self.alpha * (1 - pt) ** gamma * target * torch.log(pt) - \
                (1 - self.alpha) * pt ** gamma * (1 - target) * torch.log(1 - pt)
        
        loss /= len(self.gammas)
        
        if weight is None:
            weight = torch.ones_like(_input).to(device)
        
        loss *= weight
        
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
            
        return loss