import torch
from torch import nn
from torch.nn import functional as F
class DiceLoss(nn.Module):
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p, t = torch.sigmoid(logits).flatten(1), targets.flatten(1); smooth = 1.0
        return 1 - ((2 * (p * t).sum(1) + smooth) / (p.sum(1) + t.sum(1) + smooth)).mean()
class BCEDiceLoss(nn.Module):
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor: return .5 * F.binary_cross_entropy_with_logits(logits, targets) + .5 * DiceLoss()(logits, targets)
def build_loss(name: str) -> nn.Module:
    choices = {"dice": DiceLoss(), "bce": nn.BCEWithLogitsLoss(), "bce_dice": BCEDiceLoss()}
    if name not in choices: raise ValueError(f"Unknown loss '{name}'. Choose: {', '.join(choices)}")
    return choices[name]
