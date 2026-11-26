"""SmoovPlay Camera (framegen/vcam.py): every frame the pipeline shows goes out with its ideal time, on an even beat
whatever the network did, and the shared memory carries it to the camera. CPU only, with test_120fps's stand-ins."""
import os
import sys
import time

import numpy as np
import pytest
import torch

from framegen import vcam
from framegen.io import TICK, MultiSink, Tick
from framegen.settings import Settings
from framegen.vcam import CameraSink, Ring, StreamClock
from test_120fps import _Interp, _pipeline, _Sink

WINDOWS = sys.platform == "win32"


# ---- the clock
def _arrivals(n, rng, jitter_ms=4.0, drop=()):
    """A 60 Hz stream's ticks, each late by a random amount (exponential: mostly a little, now and then a lot, as
    Remote Play's: p95 about 3x the mean), some dropped."""
    t0 = 1000.0
    for i in range(n):
        if i in drop:
            continue
        yield i, t0 + i * TICK + rng.exponential(jitter_ms / 1000)


def test_the_clock_is_even_whatever_the_network_does():
    rng = np.random.default_rng(1)
    clock = StreamClock()
    times = [clock.tick(t, seq=i) for i, t in _arrivals(600, rng)]
    gaps = np.diff(times[60:]) * 1000
    assert abs(np.median(gaps) - 1000 * TICK) < 0.05
    # arrivals themselves vary by ~10 ms; the camera picks frames by these times, so what matters is staying
    # far under half a frame (8 ms)
    assert np.percentile(np.abs(gaps - 1000 * TICK), 99) < 1.0


def test_dropped_frames_keep_their_place():
    rng = np.random.default_rng(2)
    clock = StreamClock()
    got = {i: clock.tick(t, seq=i) for i, t in _arrivals(300, rng, drop={100, 101, 102, 200})}
    assert abs((got[103] - got[99]) - 4 * TICK) < 0.001  # numbered: the gap is known


def test_unnumbered_drops_are_caught_up_after_a_few_ticks():
    rng = np.random.default_rng(3)
    clock = StreamClock()
    out = [(i, clock.tick(t)) for i, t in _arrivals(300, rng, jitter_ms=1.0, drop={150, 151})]
    late = [(i, t) for i, t in out if i > 160]
    # once it's caught up, the ticks are where they belong again: arrival - its time stays under a tick
    for i, t in late:
        assert -0.001 < (1000.0 + i * TICK) - t + 0.0 < TICK


def test_a_steady_drift_is_followed():
    clock = StreamClock()
    period = TICK * 1.001  # the stream's clock 0.1% slow against the PC's
    times = [clock.tick(1000.0 + i * period) for i in range(3000)]
    lag = (1000.0 + 2999 * period) - times[-1]
    assert 0 <= lag < 0.005  # it keeps up, a few ms behind at most


# ---- the pipeline hands each frame its ideal time
class _IdealSink(_Sink):
    takes_ideal = True

    def __init__(self):
        super().__init__()
        self.ideal = []

    def write(self, frame, text=None, ideal=None):
        super().write(frame, text)
        self.ideal.append(ideal)


def _jittered_game(n, rng):
    """A 30 fps game on a 60 Hz stream whose ticks arrive late by varying amounts (stamped in the future, so the
    pipeline's own speed doesn't matter)."""
    t0 = time.perf_counter() + 1.0
    for i in range(n):
        yield Tick(torch.full((8, 8, 3), (i // 2) % 150, dtype=torch.uint8),
                   t0 + i * TICK + rng.exponential(0.004), i, new_hint=(i % 2 == 0), extra={"seq": i})


@pytest.mark.parametrize("hz,step", [(60, 1.0), (120, 0.5)])
def test_each_frame_shown_gets_an_even_ideal_time(hz, step):
    sink = _IdealSink()
    _pipeline(_Interp()).run(_jittered_game(240, np.random.default_rng(4)), sink, label="RIFE",
                             settings=Settings({"hz": hz, "even_out": False, "overlay": False, "audio_sync": False}))
    ideal = np.array(sink.ideal[60:])
    assert None not in sink.ideal
    gaps = np.diff(ideal) / TICK
    assert np.all(gaps > 0)  # in the order they're shown
    assert np.percentile(np.abs(gaps - step), 95) < 0.05


def test_90_fps_spaces_its_three_frames_evenly():
    sink = _IdealSink()
    _pipeline(_Interp()).run(_jittered_game(240, np.random.default_rng(5)), sink, label="RIFE",
                             settings=Settings({"hz": 90, "even_out": False, "overlay": False, "audio_sync": False}))
    gaps = np.diff(np.array(sink.ideal[60:])) / TICK
    assert np.percentile(np.abs(gaps - 2 / 3), 95) < 0.05


def test_a_sink_without_ideal_times_is_called_as_before():
    plain, ideal = _Sink(), _IdealSink()
    _pipeline(_Interp()).run(_jittered_game(60, np.random.default_rng(6)), MultiSink(plain, ideal), label="RIFE",
                             settings=Settings({"hz": 60, "even_out": False, "overlay": False, "audio_sync": False}))
    assert plain.shown == ideal.shown and len(plain.shown) == 60


# ---- test mode's bar code
def test_stamp_reads_back():
    f = np.zeros((1080, 1920, 3), np.uint8)
    for n in (0, 1, 12345, (1 << 24) - 1):
        vcam.stamp(f, n)
        assert vcam.read_stamp(f) == n


# ---- the shared memory and the sink (Windows: named shared memory)
@pytest.fixture
def ring_name():
    return f"Local\\smoovplay-camera-test-{os.getpid()}-{time.perf_counter_ns()}"


@pytest.mark.skipif(not WINDOWS, reason="the camera is Windows only")
def test_ring_round_trip(ring_name):
    w, r = Ring(ring_name), Ring(ring_name)
    w.begin(64, 32)
    r.w, r.h = 64, 32
    hdr = r.read_header()
    assert (hdr["magic"], hdr["w"], hdr["h"], hdr["format"], hdr["delay_ms"]) == (vcam.MAGIC, 64, 32, 1,
                                                                                  vcam.DELAY_MS)
    frames = [np.full((32, 64, 3), k, np.uint8) for k in range(vcam.SLOTS + 3)]
    for k, f in enumerate(frames):
        w.push(f, 1000 + k)
    assert r.read_header()["write_seq"] == len(frames)
    assert r.read(0) is None  # overwritten by now
    ideal, got = r.read(len(frames) - 1)
    assert ideal == 1000 + len(frames) - 1 and np.array_equal(got, frames[-1])
    w.close(), r.close()


@pytest.mark.skipif(not WINDOWS, reason="the camera is Windows only")
def test_the_sink_copies_only_while_the_camera_is_read(ring_name):
    st = Settings({"vcam": True})
    sink = CameraSink(st, name=ring_name)
    f = np.full((32, 64, 3), 7, np.uint8)
    sink.write(f, ideal=time.perf_counter())
    assert sink.sent == 0 and sink.idle == 1  # nobody reading: no copy
    reader = Ring(ring_name)
    reader.w, reader.h = 64, 32
    reader.beat_as_reader()
    sink.write(f, ideal=time.perf_counter())
    assert sink.sent == 1
    ideal, got = reader.read(0)
    assert np.array_equal(got, f) and ideal > 0
    st.override("vcam", False)
    sink.write(f, ideal=time.perf_counter())
    assert sink.ring is None and reader.read_header()["magic"] == 0  # off: the camera sees the writer gone
    reader.close()


def test_off_by_default_and_hdr_turns_it_off(ring_name):
    sink = CameraSink(Settings({}), name=ring_name)
    sink.write(np.zeros((32, 64, 3), np.uint8))
    assert sink.ring is None
    if WINDOWS:
        sink = CameraSink(Settings({"vcam": True}), name=ring_name)
        sink.write(np.zeros((32, 64, 3), np.float16))
        assert sink.off_why and "HDR" in sink.off_why and sink.ring is None
