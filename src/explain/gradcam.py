"""Grad-CAM (Selvaraju et al. 2017) for the imaging model, plus sanity checks on where it looks."""
from __future__ import annotations

import numpy as np
import torch
from matplotlib import colormaps
from PIL import Image

from src.models.imaging import gradcam_layer, softmax


class GradCAM:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._handle = gradcam_layer(model).register_forward_hook(self._save_activation)

    def _save_activation(self, _module, _inputs, output):
        self.activations = output.detach()
        if output.requires_grad:
            output.register_hook(self._save_gradient)

    def _save_gradient(self, grad):
        self.gradients = grad.detach()

    def remove(self) -> None:
        self._handle.remove()

    def __call__(self, batch: torch.Tensor, class_idx: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return (cam scaled to [0, 1] at input resolution, raw logits) for a single-image batch."""
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = self.model(batch)
            target = int(logits.argmax(dim=1).item()) if class_idx is None else class_idx
            logits[0, target].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = torch.nn.functional.interpolate(cam, size=batch.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        cam = cam / cam.max() if cam.max() > 0 else cam
        return cam.cpu().numpy(), logits.detach().float().cpu().numpy()


def overlay(image: Image.Image, cam: np.ndarray, alpha: float = 0.4) -> Image.Image:
    base = image.convert("RGB").resize((cam.shape[1], cam.shape[0]))
    heat = (colormaps["jet"](cam)[..., :3] * 255).astype(np.uint8)
    return Image.blend(base, Image.fromarray(heat), alpha)


def gradcam_overlay(
    image: Image.Image,
    model: torch.nn.Module,
    transform,
    device: torch.device | str = "cpu",
    temperature: float = 1.0,
    class_idx: int | None = None,
) -> dict:
    """Run the model on one slice and return calibrated probabilities, the CAM and an overlay image."""
    cam_fn = GradCAM(model)
    batch = transform(image.convert("L")).unsqueeze(0).to(device)
    try:
        cam, logits = cam_fn(batch, class_idx)
    finally:
        cam_fn.remove()
    probs = softmax(logits, temperature)[0]
    return {"probs": probs, "cam": cam, "overlay": overlay(image, cam), "class_idx": int(probs.argmax()) if class_idx is None else class_idx}


def attention_checks(image: Image.Image, cam: np.ndarray, border_fraction: float) -> dict:
    """Heuristics for 'shortcut' attention: mass in the outer border band and outside the body
    (the same body mask used in preprocessing, resized to the CAM resolution)."""
    from src.data.preprocess import body_mask

    height, width = cam.shape
    band_h, band_w = max(1, int(height * border_fraction)), max(1, int(width * border_fraction))
    border = np.zeros_like(cam, dtype=bool)
    border[:band_h, :] = border[-band_h:, :] = True
    border[:, :band_w] = border[:, -band_w:] = True
    mask_image = Image.fromarray(body_mask(image).astype(np.uint8) * 255).resize((width, height), Image.NEAREST)
    body = np.asarray(mask_image) > 0
    total = float(cam.sum()) or 1.0
    peak = np.unravel_index(int(cam.argmax()), cam.shape)
    return {
        "border_mass": float(cam[border].sum() / total),
        "border_area": float(border.mean()),
        "outside_body_mass": float(cam[~body].sum() / total),
        "peak_in_border": bool(border[peak]),
    }
