"""Minimal inference-only compatibility layer for the legacy GenDeg model."""

import torch.nn as nn

__version__ = "1.4.2-inference-shim"
LightningModule = nn.Module
LightningDataModule = object

