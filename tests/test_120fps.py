"""120 fps: the pipeline shows the 1/4, 1/2 and 3/4 frames and the real one in that order, each made once. CPU only:
a stand-in interpolator whose frames say which timestep they are."""
import time

import numpy as np
import torch

from framegen.io import Tick
from framegen.pipeline import Pipeline
from framegen.settings import Settings

# ---- the pipeline at 120 Hz, with a stand-in interpolator whose frames say which timestep they are
AI = 200  # an AI frame at t is filled with AI + round(4 * t): 201, 202, 203


class _Interp:
    h = w = 8
    device = "cpu"
    private = False

    def __init__(self):
        self.in_u8 = self.out_u8 = torch.zeros(8, 8, 3, dtype=torch.uint8)
        self.single = []

    def load(self, frame):
        self.in_u8 = frame

    def prepare(self, slot):
        pass

    def interpolate(self, slot, t=0.5):
        self.single.append(t)
        self.out_u8 = torch.full((8, 8, 3), AI + round(4 * t), dtype=torch.uint8)
        return self.out_u8


class _Sink:
    gpu = False

    def __init__(self):
        self.shown = []

    def write(self, frame, text=None):
        f = frame if torch.is_tensor(frame) else torch.from_numpy(np.asarray(frame))
        self.shown.append(int(f[0, 0, 0]))


def _pipeline(interp):
    p = Pipeline.__new__(Pipeline)  # no pinned memory: that would start CUDA
    from framegen import pipeline as pl
    p.interp, p.detector, p.cut_threshold, p.stats = interp, pl.DuplicateDetector(), 25.0, None
    p._out_pinned = torch.empty((8, 8, 3), dtype=torch.uint8)
    p._out_np = p._out_pinned.numpy()
    return p


def _game(n=120):
    """A 30 fps game on a 60 Hz stream; each tick arrives as it's taken, so none is late however slow the CPU."""
    for i in range(n):
        content = i // 2
        yield Tick(torch.full((8, 8, 3), content % 150, dtype=torch.uint8), time.perf_counter(), i,
                   new_hint=(i % 2 == 0))


def test_120_fps_shows_each_quarter_once_and_in_order():
    it, sink = _Interp(), _Sink()
    st = _pipeline(it).run(_game(), sink, label="RIFE",
                           settings=Settings({"hz": 120, "even_out": False, "overlay": False}))
    assert st.quarter_shown > 40 and st.quarter_skipped == 0
    shown = sink.shown
    first = shown.index(AI + 1)
    steady = shown[first:first + 40]
    for k in range(0, 36, 4):  # 1/4, 1/2, 3/4, then the real frame
        assert steady[k:k + 3] == [AI + 1, AI + 2, AI + 3], steady[k:k + 4]
        assert steady[k + 3] < AI
    assert set(it.single) == {0.25, 0.5, 0.75}
    assert it.single.count(0.25) == it.single.count(0.5) == it.single.count(0.75)  # each once a pair


def test_90_fps_is_unchanged():
    it, sink = _Interp(), _Sink()
    _pipeline(it).run(_game(), sink, label="RIFE", settings=Settings({"hz": 90, "even_out": False, "overlay": False}))
    assert set(it.single) == {1 / 3, 2 / 3}
