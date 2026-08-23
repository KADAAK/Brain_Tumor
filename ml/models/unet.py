"""Standard 2D U-Net and the backend-facing segmentation model protocol."""
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
import torch
from torch import nn


class BaseSegmentationModel(ABC):
    """Small contract used by the web application; implementations return a boolean mask."""
    @abstractmethod
    def predict(self, image: np.ndarray) -> np.ndarray: ...


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.layers(x)


class UNet(nn.Module):
    """Conventional U-Net. Returns logits; sigmoid is applied only for inference/metrics."""
    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_channels: int = 32) -> None:
        super().__init__()
        widths = [base_channels * (2**i) for i in range(5)]
        self.encoders = nn.ModuleList([DoubleConv(in_channels, widths[0])] + [DoubleConv(widths[i - 1], widths[i]) for i in range(1, 4)])
        self.pool = nn.MaxPool2d(2); self.bottleneck = DoubleConv(widths[3], widths[4])
        self.upconvs = nn.ModuleList([nn.ConvTranspose2d(widths[4], widths[3], 2, 2), nn.ConvTranspose2d(widths[3], widths[2], 2, 2), nn.ConvTranspose2d(widths[2], widths[1], 2, 2), nn.ConvTranspose2d(widths[1], widths[0], 2, 2)])
        self.decoders = nn.ModuleList([DoubleConv(widths[3] * 2, widths[3]), DoubleConv(widths[2] * 2, widths[2]), DoubleConv(widths[1] * 2, widths[1]), DoubleConv(widths[0] * 2, widths[0])])
        self.head = nn.Conv2d(widths[0], out_channels, 1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []
        for encoder in self.encoders:
            x = encoder(x); skips.append(x); x = self.pool(x)
        x = self.bottleneck(x)
        for upconv, decoder, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = upconv(x)
            if x.shape[-2:] != skip.shape[-2:]: x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = decoder(torch.cat([skip, x], dim=1))
        return self.head(x)
