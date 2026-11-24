"""Steady HUD (framegen/steady.py), shared by RIFE and NVIDIA VFG: where it acts, how it blends, and VFG's use of it
with a stand-in for NVIDIA's SDK (the real one is checked by hand: its results aren't for publishing)."""
from types import SimpleNamespace

import pytest
import torch

from framegen.steady import STEADY_HUD_THRESHOLD, SteadyHud

H, W = 32, 48


def _scene(shift: int) -> torch.Tensor:
    """(1, 3, H, W) RGB: a moving stripe pattern, with a still white "HUD" box (sharp edges) at rows 4-11, cols 4-19."""
    x = torch.arange(W).float()
    bg = (0.3 + 0.2 * torch.sin((x + shift) * 0.7)).expand(H, W)
    img = torch.stack([bg, bg * 0.8, bg * 0.6]).unsqueeze(0).clone()
    img[:, :, 4:12, 4:20] = 1.0
    return img


def test_it_acts_on_still_edges_only():
    hud = SteadyHud(H, W, "cpu")
    a, b = _scene(0), _scene(3)
    hud.edges(0, a)
    hud.mask(0, 1, a, b)
    w = hud.w[0, 0]
    assert w[4, 10] == 1.0 and w[11, 19] == 1.0  # on the box's edge, the same in both: the real pixels
    assert w[20, 30] == 0.0  # the moving background: the model's
    assert w[8, 12] == 0.0  # inside the box, away from its edges: the model's (a flat area isn't painted over)


def test_halo_grows_it_around_still_edges(monkeypatch):
    import framegen.steady as steady
    a, b = _scene(0), _scene(3)
    masks = {}
    for n in (0, 2):
        monkeypatch.setattr(steady, "STEADY_HUD_HALO", n)
        hud = SteadyHud(H, W, "cpu")
        hud.edges(0, a)
        hud.mask(0, 1, a, b)
        masks[n] = hud.w[0, 0]
    assert masks[0][2, 10] == 0.0 and masks[2][2, 10] == 1.0  # just outside the box: the moving scene behind it
    assert masks[2][20, 30] == 0.0  # well away from the HUD: still the model's
    assert torch.all(masks[2] >= masks[0])


def test_blend_takes_the_real_pixels_at_t():
    hud = SteadyHud(H, W, "cpu")
    a, b = _scene(0), _scene(3)
    hud.edges(0, a)
    hud.mask(0, 1, a, b)
    mid = torch.zeros_like(a)  # a model that got everything wrong
    hud.t_now.fill_(0.25)
    out = hud.blend(mid, a, b)
    assert torch.allclose(out[0, :, 4, 10], torch.lerp(a, b, 0.25)[0, :, 4, 10])
    assert torch.equal(out[0, :, 20, 30], mid[0, :, 20, 30])


def test_threshold_is_the_shared_one():
    from framegen import interpolator
    assert interpolator.STEADY_HUD_THRESHOLD == STEADY_HUD_THRESHOLD  # still importable from where it was


class _FakeVfg:
    """NVIDIA VFG stand-in: the "AI frame" is the newest real frame shifted 2 pixels right, so a still HUD wobbles."""

    def __init__(self, **kw):
        pass

    def load(self):
        pass

    def run_at_timestep(self, prev, cur, t, stream_ptr=None):
        return SimpleNamespace(image=torch.roll(cur.view(torch.int32), 2, dims=1).contiguous())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="VfgInterpolator runs on the GPU")
def test_vfg_with_steady_hud_keeps_the_hud_still(monkeypatch):
    from framegen import vfg
    fake = SimpleNamespace(ImageEncoding=SimpleNamespace(RGB10A2=0, RGB8=1), VideoFrameGeneration=_FakeVfg)
    monkeypatch.setattr(vfg, "import_vfx", lambda: fake)
    it = vfg.VfgInterpolator(W, H, encoding="rgb10a2")

    def frame(shift):  # the pipeline's frames: HxWx3 BGR uint8 on the GPU
        return (_scene(shift)[0].flip(0).permute(1, 2, 0) * 255).round().to(torch.uint8).cuda()

    for slot, shift in ((0, 0), (1, 3)):
        it.load(frame(shift))
        it.prepare(slot)
    hud_px = (slice(5, 11), slice(4, 6))  # just inside the box's left edge, which the stand-in moves
    off = it.interpolate(1, 0.5).clone()
    it.set_steady(STEADY_HUD_THRESHOLD)  # turned on between two real frames: taken from VFG's own copies
    on = it.interpolate(1, 0.5).clone()
    real = frame(0)
    assert not torch.equal(off[hud_px], real[hud_px])  # the stand-in dragged the box's edge
    assert torch.equal(on[hud_px], real[hud_px])  # Steady HUD put it back
    assert torch.equal(on[20, 30:40], off[20, 30:40])  # the moving background is the model's either way
    it.set_steady(0.0)
    assert torch.equal(it.interpolate(1, 0.5), off)  # off again: VFG's frame as it was
    it.load(frame(6))  # and with it on, the next real frame's mask comes with prepare()
    it.set_steady(STEADY_HUD_THRESHOLD)
    it.prepare(0)
    assert torch.equal(it.interpolate(0, 0.5)[hud_px], real[hud_px])
