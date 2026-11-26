"""SmoovPlay Camera: the pipeline's frames for OBS, as a webcam ("Video Capture Device"), on an even beat.

OBS's Window Capture takes a picture of SmoovPlay's window every 1/60 s. At 60 fps the in-between frame is often up
for only a few ms (it's ready late; the real frame follows straight after), so most in-between frames never reach
the stream, and a fullscreen window that's presented straight to the display can't be captured at all. The camera
doesn't look at the window: every frame the pipeline shows goes, with its *ideal* time, into shared memory, and the
camera (native/vcam, a DirectShow filter OBS loads) hands them to OBS on an even 60 fps beat, a fixed delay D later.
No menu, message or stats overlay: those are drawn on the window only.

The ideal time of a frame: the pipeline's ticks come from a 60 Hz stream, so tick n's time is base + n * period on a
clock fitted to their arrivals (StreamClock: network jitter doesn't move it, drift does), plus the frame's place
within its tick at 90 and 120 fps (pipeline.py, `present(..., at=)`).

Shared memory "Local\\smoovplay-camera" (SPVC, little-endian), HEADER bytes, then SLOTS slots of SLOT_STRIDE:

    0  u32 magic 'SPVC' (0 while the writer is (re)starting or gone)   4  u32 version (1)
    8  u32 slot_count     12 u32 slot_stride     16 u32 width     20 u32 height
    24 u32 format (1: BGR, 3 bytes a pixel, rows top to bottom, packed)
    28 u32 fps (60, what the camera delivers)   32 u32 delay_ms (D)   36 u32 session (+1 per start)
    40 u64 write_seq (frames written; the next goes in slot write_seq % slot_count)
    48 u64 qpc_freq       56 u64 writer heartbeat (QPC)   64 u64 reader heartbeat (QPC, the camera writes it)
    72 u64 frames delivered  80 u64 repeats  88 u64 skipped (the camera's counts, for the log)
    slot: 0 u64 seq (all ones while it's being written)  8 u64 ideal (QPC)  16 u64 written (QPC)  then the frame

Nobody reading (no reader heartbeat for READER_GONE_S): frames aren't copied at all, so the camera costs nothing
until OBS opens it.
"""
from __future__ import annotations

import os
import struct
import sys
import time

import numpy as np

NAME = "Local\\smoovplay-camera"
MAGIC = 0x43565053  # 'SPVC'
VERSION = 1
FORMAT_BGR = 1
HEADER = 4096
SLOT_HEADER = 64
SLOTS = 8  # D (DELAY_MS) of frames at 120 fps, and some
MAX_W, MAX_H = 2560, 1440  # up to a 1440p capture; 4K is too big for the ring
SLOT_STRIDE = -(-(SLOT_HEADER + MAX_W * MAX_H * 3) // 4096) * 4096
TOTAL = HEADER + SLOTS * SLOT_STRIDE
FPS = 60
DELAY_MS = 80  # D: frames go to OBS this long after their ideal time (covers the pipeline's own delay, p99)
READER_GONE_S = 2.0
U64_MAX = (1 << 64) - 1
# Test mode: the frame's number as a bar code in the top-left corner of the camera's copy (never the window's), so a
# recording can be checked for repeated and skipped frames (scripts/vcam_check.py)
STAMP = os.environ.get("SMOOVPLAY_CAMERA_STAMP") == "1"
STAMP_BITS, STAMP_CELL = 24, 8


class StreamClock:
    """Even times for a 60 Hz stream's ticks: base + n * period, fitted to when they arrived.

    Arrivals are late by a varying amount (the network, decoding) but never early, so the clock follows an arrival
    that's earlier than it expected at once, and one that's later only slowly (DRIFT_GAIN): a late frame leaves it
    where it was, while a steady drift between the stream's clock and the PC's moves it. n comes from the source's
    own frame numbers when it has them (frames the queue dropped still count), else one a tick; ticks running a whole
    period or more late for a while (frames dropped that weren't numbered) move n on."""

    DRIFT_GAIN = 0.01
    SKIP_TICKS = 3

    def __init__(self, period: float = 1 / 60) -> None:
        self.period = period
        self.reset()

    def reset(self) -> None:
        self.base: float | None = None
        self.n = 0
        self._seq: int | None = None
        self._late = 0

    def tick(self, t_arrival: float, seq: int | None = None) -> float:
        """The even time of this tick (perf_counter seconds)."""
        if self.base is None:
            self.base, self.n, self._seq, self._late = t_arrival, 0, seq, 0
            return t_arrival
        if seq is not None and self._seq is not None and seq > self._seq:
            self.n += min(seq - self._seq, 600)
        else:
            self.n += 1
        self._seq = seq
        e = t_arrival - (self.base + self.n * self.period)
        if e < 0:
            self.base += e  # earlier than it thought: the clock was late
            self._late = 0
        elif e >= 0.9 * self.period:
            self._late += 1
            if self._late >= self.SKIP_TICKS:  # running a tick or more behind: frames were dropped unnumbered
                self.n += int(e / self.period + 0.5)
                self._late = 0
                return self.tick_time()
        else:
            self._late = 0
        self.base += e * self.DRIFT_GAIN if e > 0 else 0.0
        return self.tick_time()

    def tick_time(self) -> float:
        return self.base + self.n * self.period


def stamp(frame: np.ndarray, number: int) -> None:
    """Test mode: number's low STAMP_BITS bits as black/white cells along the top-left (in place)."""
    for b in range(STAMP_BITS):
        frame[:STAMP_CELL, b * STAMP_CELL:(b + 1) * STAMP_CELL] = 255 if (number >> b) & 1 else 0


def read_stamp(frame: np.ndarray) -> int:
    """stamp()'s number back, from a frame (BGR, possibly scaled and compressed: cells read by their centres)."""
    h, w = frame.shape[:2]
    sx, sy = w / 1920 if w != 1920 else 1.0, h / 1080 if h != 1080 else 1.0
    y = int(STAMP_CELL * sy / 2)
    n = 0
    for b in range(STAMP_BITS):
        x = int((b + 0.5) * STAMP_CELL * sx)
        if float(np.mean(frame[y, x])) > 127:
            n |= 1 << b
    return n


class Ring:
    """The shared memory, writer's side (the reader's side for tests: read())."""

    def __init__(self, name: str = NAME) -> None:
        from .tapsys import open_map, qpc, qpc_freq
        self._qpc = qpc
        self.freq = qpc_freq()
        self.mm = open_map(name, TOTAL)
        # perf_counter -> QPC (both the same counter on Windows; taken together once)
        self._pc0, self._q0 = time.perf_counter(), qpc()

    def to_qpc(self, t: float) -> int:
        return self._q0 + int(round((t - self._pc0) * self.freq))

    def begin(self, w: int, h: int, delay_ms: int = DELAY_MS) -> None:
        struct.pack_into("<I", self.mm, 0, 0)
        struct.pack_into("<7I", self.mm, 4, VERSION, SLOTS, SLOT_STRIDE, w, h, FORMAT_BGR, FPS)
        (session,) = struct.unpack_from("<I", self.mm, 36)
        struct.pack_into("<II", self.mm, 32, delay_ms, session + 1)
        struct.pack_into("<QQQ", self.mm, 40, 0, self.freq, self._qpc())
        for i in range(SLOTS):
            struct.pack_into("<Q", self.mm, HEADER + i * SLOT_STRIDE, U64_MAX)
        struct.pack_into("<I", self.mm, 0, MAGIC)
        self.w, self.h = w, h

    def reader_alive(self) -> bool:
        (beat,) = struct.unpack_from("<Q", self.mm, 64)
        return beat != 0 and (self._qpc() - beat) / self.freq < READER_GONE_S

    def push(self, frame: np.ndarray, ideal_qpc: int) -> None:
        """One frame (h x w x 3 uint8, the size begin() was given)."""
        (seq,) = struct.unpack_from("<Q", self.mm, 40)
        off = HEADER + (seq % SLOTS) * SLOT_STRIDE
        struct.pack_into("<Q", self.mm, off, U64_MAX)
        dst = np.frombuffer(self.mm, np.uint8, self.w * self.h * 3, off + SLOT_HEADER).reshape(self.h, self.w, 3)
        np.copyto(dst, frame)
        if STAMP:
            stamp(dst, seq)
        now = self._qpc()
        struct.pack_into("<QQ", self.mm, off + 8, ideal_qpc, now)
        struct.pack_into("<Q", self.mm, off, seq)
        struct.pack_into("<QQ", self.mm, 40, seq + 1, self.freq)
        struct.pack_into("<Q", self.mm, 56, now)

    def heartbeat(self) -> None:
        struct.pack_into("<Q", self.mm, 56, self._qpc())

    def counts(self) -> tuple[int, int, int]:
        """The camera's own counts: frames delivered, repeated, skipped."""
        return struct.unpack_from("<QQQ", self.mm, 72)

    def end(self) -> None:
        struct.pack_into("<I", self.mm, 0, 0)

    def close(self) -> None:
        self.mm.close()

    # ---- the reader's side, as the camera does it (tests, scripts/vcam_check.py --ring)
    def read_header(self) -> dict:
        magic, version, n, stride, w, h, fmt, fps, delay, session = struct.unpack_from("<10I", self.mm, 0)
        write_seq, freq = struct.unpack_from("<QQ", self.mm, 40)
        return dict(magic=magic, version=version, slots=n, stride=stride, w=w, h=h, format=fmt, fps=fps,
                    delay_ms=delay, session=session, write_seq=write_seq, freq=freq)

    def read(self, seq: int) -> tuple[int, np.ndarray] | None:
        """Frame seq and its ideal time, or None if it's been overwritten (or is being written)."""
        off = HEADER + (seq % SLOTS) * SLOT_STRIDE
        (s, ideal) = struct.unpack_from("<QQ", self.mm, off)
        if s != seq:
            return None
        frame = np.frombuffer(self.mm, np.uint8, self.w * self.h * 3, off + SLOT_HEADER).reshape(self.h, self.w,
                                                                                                 3).copy()
        return (ideal, frame) if struct.unpack_from("<Q", self.mm, off)[0] == seq else None

    def beat_as_reader(self) -> None:
        struct.pack_into("<Q", self.mm, 64, self._qpc())


class CameraSink:
    """A MultiSink member (framegen/io.py) that feeds SmoovPlay Camera while settings.vcam is on. Takes each frame
    with its ideal time (takes_ideal); copies it only while the camera is being read (OBS has it open)."""

    takes_ideal = True
    gpu = False  # it copies to the CPU itself

    def __init__(self, settings, name: str = NAME) -> None:
        self.settings, self.name = settings, name
        self.ring: Ring | None = None
        self.sent = self.idle = 0
        self.late_ms: list[float] = []  # how long after its ideal time each frame was written (vs DELAY_MS)
        self.off_why: str | None = None  # why it can't run (said once)
        self._reading = False

    def _on(self) -> bool:
        return sys.platform == "win32" and bool(getattr(self.settings, "vcam", False)) and self.off_why is None

    def write(self, frame, text: str | None = None, ideal: float | None = None):
        if not self._on():
            if self.ring is not None:
                self.close()
            return None
        h, w = frame.shape[:2]
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            return self._cant("SmoovPlay Camera: HDR output isn't supported yet; the camera is off")
        if w > MAX_W or h > MAX_H:
            return self._cant(f"SmoovPlay Camera: {w}x{h} is bigger than it takes ({MAX_W}x{MAX_H}); the camera is off")
        if self.ring is None or (self.ring.w, self.ring.h) != (w, h):
            if self.ring is None:
                self.ring = Ring(self.name)
            self.ring.begin(w, h)
            print(f"[camera] on: {w}x{h} at {FPS} fps, {DELAY_MS} ms behind the game (in OBS: Video Capture Device "
                  f"'SmoovPlay Camera', Buffering Enable; the game's sound gets a sync offset of +{DELAY_MS} ms)",
                  flush=True)
        if not self.ring.reader_alive():
            if self._reading:
                print("[camera] OBS stopped reading SmoovPlay Camera", flush=True)
                self._reading = False
            self.idle += 1
            self.ring.heartbeat()
            return None
        if not self._reading:
            print("[camera] OBS is reading SmoovPlay Camera", flush=True)
            self._reading = True
        if ideal is None:
            ideal = time.perf_counter()
        host = frame.cpu().numpy() if hasattr(frame, "cpu") else frame
        self.ring.push(host, self.ring.to_qpc(ideal))
        self.sent += 1
        self.late_ms.append((time.perf_counter() - ideal) * 1000)
        return None

    def _cant(self, why: str):
        self.off_why = why
        print(f"[camera] {why}", flush=True)
        if self.ring is not None:
            self.close()
        return None

    def summary(self) -> str:
        if not self.sent and not self.idle:
            return ""
        line = f"SmoovPlay Camera: {self.sent} frames sent"
        if self.late_ms:
            lm = np.array(self.late_ms)
            line += (f", written {np.median(lm):.0f} ms after their time (p99 {np.percentile(lm, 99):.0f}; "
                     f"{np.mean(lm > DELAY_MS) * 100:.1f}% later than the camera's {DELAY_MS} ms)")
        if self.ring is not None:
            delivered, repeats, late = self.ring.counts()
            if delivered:
                line += f"; the camera delivered {delivered}: {repeats} repeats, {late} frames skipped"
        return line

    def close(self) -> None:
        if self.ring is not None:
            line = self.summary()
            if line:
                print(f"[camera] {line}", flush=True)
            self.ring.end()
            self.ring.close()
            self.ring = None
        self._reading = False
