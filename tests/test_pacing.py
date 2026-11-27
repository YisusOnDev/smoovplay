"""Even pacing (framegen/pacing.py): frames on screen on a fixed beat, in real time. A stand-in interpolator that
takes a set time, a 30 fps game on a 60 Hz stream whose frames arrive with jitter, and a sink noting when each frame
was shown."""
import random
import time

import numpy as np
import torch

from framegen.io import TICK, Tick
from framegen.pacing import EvenPacer
from framegen.settings import Settings
from test_120fps import _pipeline


class _SlowInterp:
    h = w = 8
    device = "cpu"
    private = False

    def __init__(self, model_s: float):
        self.model_s = model_s
        self.in_u8 = torch.zeros(8, 8, 3, dtype=torch.uint8)

    def load(self, frame):
        self.in_u8 = frame

    def prepare(self, slot):
        pass

    def interpolate(self, slot, t=0.5):
        end = time.perf_counter() + self.model_s
        while time.perf_counter() < end:
            pass
        return torch.full((8, 8, 3), 200, dtype=torch.uint8)


class _TimedSink:
    gpu = False

    def __init__(self):
        self.at = []

    def write(self, frame, text=None):
        self.at.append(time.perf_counter())


def _stream(n: int, jitter_ms: float, seed: int = 1):
    """Ticks arriving on a 60 Hz beat, each late by up to jitter_ms (the network); a tick is handed over when it has
    arrived, so a slow pipeline finds it waiting, as with a real queue."""
    rnd = random.Random(seed)
    t0 = time.perf_counter() + 0.05
    for i in range(n):
        arrive = t0 + i * TICK + rnd.uniform(0, jitter_ms) / 1000
        while time.perf_counter() < arrive:
            pass
        yield Tick(torch.full((8, 8, 3), (i // 2) % 150, dtype=torch.uint8), arrive, i, new_hint=(i % 2 == 0))


def _gaps_ms(model_ms: float, paced: bool, n: int = 240, jitter_ms: float = 3.0):
    sink = _TimedSink()
    st = _pipeline(_SlowInterp(model_ms / 1000)).run(
        _stream(n, jitter_ms), sink, label="RIFE",
        settings=Settings({"hz": 60, "even_out": False, "overlay": False, "even_pacing": paced}))
    gaps = np.diff(sink.at[n // 2:]) * 1000  # the second half: D measured by then
    return gaps, st


def test_paced_frames_are_evenly_spaced():
    off, _ = _gaps_ms(8, paced=False)
    on, st = _gaps_ms(8, paced=True)
    # Without it, the in-between frame is up for about the model's time and the real one for the rest of the pair
    assert np.mean(np.abs(off - 1000 * TICK)) > 4, off
    assert np.percentile(np.abs(on - 1000 * TICK), 90) < 2.5, on
    assert "even pacing" in st.pacing


def test_a_model_slower_than_a_tick_gets_closer_without_falling_behind():
    """22 ms for an in-between frame: the frame after it can't wait for its even time without delaying the next
    in-between frame, so it's shown at the next tick's time instead (the cap). Closer to even, and no backlog."""
    off, _ = _gaps_ms(22, paced=False)
    on, st = _gaps_ms(22, paced=True)
    spread = lambda g: np.mean(np.abs(g - 1000 * TICK))  # noqa: E731
    assert spread(on) < spread(off) - 2, (spread(on), spread(off))
    assert max(st.queue_ms[-40:]) < 1000 * TICK, st.queue_ms[-40:]  # not falling further and further behind


def test_d_follows_the_95th_percentile_and_shrinks_slowly():
    p = EvenPacer()
    ideal = 100.0
    for k in range(240):
        p.due(ideal + k * TICK, ideal + k * TICK + 0.010, cap=ideal + (k + 1) * TICK)
    assert abs(p.d - 0.0115) < 0.0005  # 10 ms ready + the margin
    for k in range(240, 300):
        p.due(ideal + k * TICK, ideal + k * TICK + 0.002, cap=ideal + (k + 1) * TICK)
    assert p.d > 0.0115 - 3 * EvenPacer.SHRINK  # a calm second doesn't undo it at once
    # never past the next tick's time, never before now
    assert p.due(200.0, 200.001, cap=200.0 + TICK) <= 200.0 + TICK
    assert p.due(300.0, 300.05, cap=300.0 + TICK) == 300.05
