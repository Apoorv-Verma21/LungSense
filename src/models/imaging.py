"""Imaging model construction, loading and calibrated prediction."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.utils.common import cfg_path


def build_model(backbone: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    from torchvision import models

    if backbone == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif backbone == "densenet121":
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
    return model


def gradcam_layer(model: nn.Module) -> nn.Module:
    """Last convolutional block, used for Grad-CAM and embeddings."""
    return model.features[-1]


def embed(model: nn.Module, batch: torch.Tensor) -> torch.Tensor:
    """Pooled penultimate features, matching each backbone's own forward pass."""
    from torchvision.models import DenseNet

    features = model.features(batch)
    if isinstance(model, DenseNet):
        features = torch.relu(features)
    return torch.flatten(nn.functional.adaptive_avg_pool2d(features, 1), 1)


def load_trained(cfg: dict, device: torch.device | str = "cpu") -> tuple[nn.Module, float]:
    """Load the best checkpoint and its validation-fitted temperature."""
    model_file = cfg_path(cfg, "model_file")
    if not model_file.exists():
        raise FileNotFoundError(f"{model_file} is missing - run scripts/03_train_imaging.py first.")
    checkpoint = torch.load(model_file, map_location=device, weights_only=False)
    model = build_model(checkpoint["backbone"], len(checkpoint["classes"]), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, float(checkpoint.get("temperature", 1.0))


@torch.no_grad()
def predict_logits(model: nn.Module, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits, labels = [], []
    for images, targets in loader:
        logits.append(model(images.to(device)).float().cpu())
        labels.append(targets)
    return torch.cat(logits).numpy(), torch.cat(labels).numpy()


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = logits / temperature
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Temperature scaling (Guo et al. 2017), fit on the validation split only."""
    logits_t = torch.tensor(logits, dtype=torch.float64)
    labels_t = torch.tensor(labels, dtype=torch.long)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(logits_t / log_t.exp(), labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.exp().item())


def model_path_exists(cfg: dict) -> bool:
    return Path(cfg_path(cfg, "model_file")).exists()
