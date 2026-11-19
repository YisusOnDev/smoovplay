"""Steady HUD (settings "steady_hud"), for any interpolation model: RIFE (framegen/interpolator.py, inside its
captured CUDA graphs) and NVIDIA VFG (framegen/vfg.py).

Where a pixel is (almost) the same in both real frames, like a map, health bar or subtitle over a moving scene, the AI
frame takes the real pixel (blended at t, so fades still fade) instead of the model's, which can drag it along with
the motion behind it. The edge is soft: fully real below half the threshold, fully the model's above it, so there's
no speckle where the difference hovers near it.

Only near edges (within STEADY_HUD_EDGE_R pixels of a brightness step of about 32 levels, in either real frame):
text, icons and map lines, where wobble shows. On a flat background the model's pixel is kept, so a pole or a car
crossing a sky or road that matches before and after isn't painted over. On eight 60 fps gameplay clips
(scripts/steady_hud_eval.py) that keeps 96% of the fix on HUD-like pixels at a quarter of the cost elsewhere.

Where it acts depends only on the two real frames, so mask() works it out once per real frame and each in-between
frame just blends (blend()).
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F

# The largest difference between the two real frames (0..1 per colour channel) at which a pixel still counts as
# unchanged. 0.03 is about 8 of 255 levels: above Remote Play's compression noise on a still HUD.
STEADY_HUD_THRESHOLD = float(os.environ.get("FRAMEGEN_STEADY_THRESHOLD", "0.03"))  # env: for tuning only
# ...and only within this many pixels of an edge: Sobel L1 of brightness (0..1) above STEADY_HUD_EDGE, about a
# 32-level step.
STEADY_HUD_EDGE = 128.0 / 255.0
STEADY_HUD_EDGE_R = 1
# Then grown by this many pixels, so the ring around a still glyph (its anti-aliased edge and outline, which change
# with the scene behind them) gets the real pixels blended at t as well, instead of the model's, which drags the text's
# edge along with the motion. 0 keeps the mask as it was.
STEADY_HUD_HALO = int(os.environ.get("FRAMEGEN_STEADY_HALO", "0"))  # env: for trying it out


def halo(w: torch.Tensor, n: int | None = None) -> torch.Tensor:
    """The mask w (1, 1, h, w) grown by n pixels (STEADY_HUD_HALO by default)."""
    n = STEADY_HUD_HALO if n is None else n
    return F.max_pool2d(w, 2 * n + 1, 1, n) if n > 0 else w


class SteadyHud:
    """Steady HUD's state for one interpolator. Real frames are (1, 3, h, w) RGB, 0..1 (HDR output's PQ values as
    they are). The tensors here stay put, so a CUDA graph can read them; t_now and still_thr change in place."""

    def __init__(self, h: int, w: int, device) -> None:
        self.t_now = torch.full((1, 1, 1, 1), 0.5, device=device)
        self.still_thr = torch.full((1, 1, 1, 1), STEADY_HUD_THRESHOLD, device=device)
        gx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=device)
        self.sobel = torch.stack([gx, gx.t()]).unsqueeze(1).half()  # (2, 1, 3, 3): x and y gradients
        self.luma = torch.tensor([0.299, 0.587, 0.114], device=device).view(1, 3, 1, 1).half()  # RGB
        self.edge = [torch.zeros((1, 1, h, w), dtype=torch.half, device=device) for _ in range(2)]  # per real frame
        self.w = torch.zeros((1, 1, h, w), device=device)  # 1: the real pixels, 0: the model's

    @torch.no_grad()
    def edges(self, slot: int, rgb: torch.Tensor) -> None:
        """edge[slot]: 1 on a brightness edge of that real frame (once per frame, in half precision)."""
        lum = F.conv2d(rgb.half(), self.luma)
        g = F.conv2d(lum, self.sobel, padding=1).abs_().sum(1, keepdim=True)
        self.edge[slot].copy_(g > STEADY_HUD_EDGE)

    @torch.no_grad()
    def mask(self, a: int, b: int, rgb_a: torch.Tensor, rgb_b: torch.Tensor) -> None:
        """Where it acts between real frames a -> b (slots; b the newest). The newest frame's edges are worked out
        here; the older one's were, when it was the newest."""
        self.edges(b, rgb_b)
        ia, ib = rgb_a.half(), rgb_b.half()
        d = (ib - ia).abs_().amax(1, keepdim=True)
        thr = self.still_thr.half()
        w = ((thr - d) * (2.0 / thr)).clamp_(0.0, 1.0)
        r = STEADY_HUD_EDGE_R
        self.w.copy_(halo(w * F.max_pool2d(torch.maximum(self.edge[a], self.edge[b]), 2 * r + 1, 1, r)))

    def blend(self, mid: torch.Tensor, rgb_a: torch.Tensor, rgb_b: torch.Tensor) -> torch.Tensor:
        """The AI frame `mid` (1, 3, h, w) with the real pixels, at t_now, where mask() said."""
        return torch.lerp(mid.float(), torch.lerp(rgb_a.float(), rgb_b.float(), self.t_now), self.w)
