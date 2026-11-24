"""Match the monitor's refresh rate to SmoovPlay's output while it's fullscreen.

60 fps on a 144 Hz monitor with VSync puts each frame up for 2 or 3 refreshes in turn (144 / 60 = 2.4): an uneven
beat however even the frames are. A refresh rate that's a whole multiple of the output (120 Hz for 60 fps) shows
every frame for the same time.

The change is Windows' temporary kind (ChangeDisplaySettingsEx with CDS_FULLSCREEN): it isn't saved, and Windows puts
the desktop's own rate back when SmoovPlay closes, even if it crashes. Leaving fullscreen puts it back sooner.
"""
from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes as wt

DM_PELSWIDTH, DM_PELSHEIGHT, DM_DISPLAYFREQUENCY = 0x80000, 0x100000, 0x400000
CDS_FULLSCREEN = 0x4
DISP_CHANGE_SUCCESSFUL = 0
ENUM_CURRENT_SETTINGS = 0xFFFFFFFF


def pick(out_hz: int, current: int, available) -> int | None:
    """The refresh rate to switch to for `out_hz` frames a second, or None to stay at `current`.

    The fastest rate in `available` (the monitor's rates at its current resolution) that's a whole multiple of
    out_hz: 120 rather than 60 for 60 fps, as a faster refresh also waits less for VSync. None when `current` is
    already a multiple and as fast, or when no rate is one (90 fps on a 60/120/144 Hz monitor: nothing to gain)."""
    if out_hz <= 0:
        return None
    fits = [r for r in set(available) | {current} if r >= out_hz and r % out_hz == 0]
    if not fits:
        return None
    best = max(fits)
    return None if best == current else best


class _DEVMODE(ctypes.Structure):  # DEVMODEW, display fields
    _fields_ = [("dmDeviceName", wt.WCHAR * 32), ("dmSpecVersion", wt.WORD), ("dmDriverVersion", wt.WORD),
                ("dmSize", wt.WORD), ("dmDriverExtra", wt.WORD), ("dmFields", wt.DWORD),
                ("dmPositionX", wt.LONG), ("dmPositionY", wt.LONG), ("dmDisplayOrientation", wt.DWORD),
                ("dmDisplayFixedOutput", wt.DWORD), ("dmColor", ctypes.c_short), ("dmDuplex", ctypes.c_short),
                ("dmYResolution", ctypes.c_short), ("dmTTOption", ctypes.c_short), ("dmCollate", ctypes.c_short),
                ("dmFormName", wt.WCHAR * 32), ("dmLogPixels", wt.WORD), ("dmBitsPerPel", wt.DWORD),
                ("dmPelsWidth", wt.DWORD), ("dmPelsHeight", wt.DWORD), ("dmDisplayFlags", wt.DWORD),
                ("dmDisplayFrequency", wt.DWORD), ("dmICMMethod", wt.DWORD), ("dmICMIntent", wt.DWORD),
                ("dmMediaType", wt.DWORD), ("dmDitherType", wt.DWORD), ("dmReserved1", wt.DWORD),
                ("dmReserved2", wt.DWORD), ("dmPanningWidth", wt.DWORD), ("dmPanningHeight", wt.DWORD)]


def _u32():
    u32 = ctypes.windll.user32
    u32.EnumDisplaySettingsW.argtypes = [wt.LPCWSTR, wt.DWORD, ctypes.POINTER(_DEVMODE)]
    u32.ChangeDisplaySettingsExW.argtypes = [wt.LPCWSTR, ctypes.POINTER(_DEVMODE), wt.HWND, wt.DWORD,
                                             ctypes.c_void_p]
    u32.ChangeDisplaySettingsExW.restype = ctypes.c_long
    return u32


def _mode(u32, device: str, index: int) -> _DEVMODE | None:
    dm = _DEVMODE()
    dm.dmSize = ctypes.sizeof(dm)
    return dm if u32.EnumDisplaySettingsW(device, index, ctypes.byref(dm)) else None


def rates(device: str) -> tuple[int, int, int, list[int]] | None:
    """The display's current width, height and refresh rate, and every rate it offers at that size; None if Windows
    won't say."""
    u32 = _u32()
    now = _mode(u32, device, ENUM_CURRENT_SETTINGS)
    if now is None or now.dmDisplayFrequency <= 1:
        return None
    w, h, found, i = now.dmPelsWidth, now.dmPelsHeight, set(), 0
    while (m := _mode(u32, device, i)) is not None:
        if (m.dmPelsWidth, m.dmPelsHeight) == (w, h) and m.dmDisplayFrequency > 1:
            found.add(int(m.dmDisplayFrequency))
        i += 1
    return int(w), int(h), int(now.dmDisplayFrequency), sorted(found)


class RefreshMatch:
    """Switches one display's refresh rate for SmoovPlay's output, and puts it back. Does nothing off Windows."""

    def __init__(self) -> None:
        self.device: str | None = None  # the display switched, while it is
        self.was = 0  # its rate before

    def match(self, device: str, out_hz: int) -> str:
        """Switch `device` to the best rate for out_hz (see pick); a line for the log, "" if nothing changed."""
        if sys.platform != "win32" or not device:
            return ""
        if self.device is not None and self.device != device:
            self.restore()
        info = rates(device)
        if info is None:
            return ""
        w, h, current, available = info
        want = pick(out_hz, current, available)
        if want is None:
            return ""
        dm = _DEVMODE()
        dm.dmSize = ctypes.sizeof(dm)
        dm.dmPelsWidth, dm.dmPelsHeight, dm.dmDisplayFrequency = w, h, want
        dm.dmFields = DM_PELSWIDTH | DM_PELSHEIGHT | DM_DISPLAYFREQUENCY
        r = _u32().ChangeDisplaySettingsExW(device, ctypes.byref(dm), None, CDS_FULLSCREEN, None)
        if r != DISP_CHANGE_SUCCESSFUL:
            return f"[display] couldn't switch {device} to {want} Hz for {out_hz} fps (error {r}); staying at {current} Hz"
        if self.device is None:
            self.device, self.was = device, current
        return f"[display] {device}: {current} Hz -> {want} Hz while fullscreen, to match {out_hz} fps"

    def restore(self) -> str:
        """Put the display back to its own (saved) mode; a line for the log, "" if it wasn't switched."""
        if self.device is None:
            return ""
        device, was = self.device, self.was
        self.device, self.was = None, 0
        _u32().ChangeDisplaySettingsExW(device, None, None, 0, None)
        return f"[display] {device}: back to {was} Hz"
