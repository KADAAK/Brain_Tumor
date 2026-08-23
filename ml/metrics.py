"""Segmentation metrics; classification accuracy is intentionally omitted."""
import torch
@torch.no_grad()
def segmentation_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float = .5, epsilon: float = 1e-7) -> dict[str, float]:
    p, t = (torch.sigmoid(logits) >= threshold).float(), (targets >= .5).float(); dims = tuple(range(1, p.ndim))
    tp, fp, fn = (p*t).sum(dims), (p*(1-t)).sum(dims), ((1-p)*t).sum(dims)
    return {"dice": float(((2*tp+epsilon)/(2*tp+fp+fn+epsilon)).mean()), "iou": float(((tp+epsilon)/(tp+fp+fn+epsilon)).mean()), "precision": float(((tp+epsilon)/(tp+fp+epsilon)).mean()), "recall": float(((tp+epsilon)/(tp+fn+epsilon)).mean())}
