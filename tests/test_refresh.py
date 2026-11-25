"""Match refresh rate: which rate fullscreen switches the monitor to (framegen/refresh.py), and when it's put back."""
from types import SimpleNamespace

from framegen import refresh
from framegen.refresh import RefreshMatch, pick


def test_60_fps_on_144_hz_goes_to_120():
    assert pick(60, 144, [60, 100, 120, 144]) == 120


def test_the_fastest_whole_multiple_wins():
    assert pick(60, 144, [60, 120, 144, 240]) == 240
    assert pick(120, 144, [60, 120, 144]) == 120


def test_already_fitting_stays():
    assert pick(60, 120, [60, 120, 144]) is None
    assert pick(60, 60, [60]) is None
    assert pick(120, 240, [60, 120, 240]) is None


def test_nothing_fits_stays():
    assert pick(90, 144, [60, 120, 144]) is None
    assert pick(60, 144, [144]) is None
    assert pick(0, 144, [60, 144]) is None


def test_90_fps_takes_180_when_the_monitor_has_it():
    assert pick(90, 144, [60, 90, 120, 144, 180]) == 180


class _User32:
    """Stand-in for Windows' display mode calls: one monitor at 144 Hz offering 60/120/144."""

    def __init__(self):
        self.calls = []

    def ChangeDisplaySettingsExW(self, device, dm, hwnd, flags, param):
        self.calls.append((device, None if dm is None else dm._obj.dmDisplayFrequency, flags))
        return 0


def test_switches_then_puts_back(monkeypatch):
    u32 = _User32()
    monkeypatch.setattr(refresh.sys, "platform", "win32")
    monkeypatch.setattr(refresh, "_u32", lambda: u32)
    monkeypatch.setattr(refresh, "rates", lambda device: (2560, 1440, 144, [60, 120, 144]))
    m = RefreshMatch()
    assert "144 Hz -> 120 Hz" in m.match(r"\\.\DISPLAY1", 60)
    assert u32.calls == [(r"\\.\DISPLAY1", 120, refresh.CDS_FULLSCREEN)]
    assert "back to 144 Hz" in m.restore()
    assert u32.calls[-1] == (r"\\.\DISPLAY1", None, 0)  # the saved mode
    assert m.restore() == ""  # once


def test_output_change_keeps_the_desktops_own_rate_to_go_back_to(monkeypatch):
    u32, now = _User32(), {"hz": 144}
    monkeypatch.setattr(refresh.sys, "platform", "win32")
    monkeypatch.setattr(refresh, "_u32", lambda: u32)
    monkeypatch.setattr(refresh, "rates", lambda device: (1920, 1080, now["hz"], [60, 120, 144, 240]))
    m = RefreshMatch()
    m.match("D", 60)  # -> 240
    now["hz"] = 240
    assert m.match("D", 120) == ""  # 240 fits 120 too
    assert "back to 144 Hz" in m.restore()


def test_not_on_other_systems(monkeypatch):
    monkeypatch.setattr(refresh.sys, "platform", "darwin")
    assert RefreshMatch().match("D", 60) == ""


def test_the_window_matches_in_fullscreen_only(monkeypatch):
    from framegen.io import GLWindowSink
    sink = object.__new__(GLWindowSink)
    seen = []
    sink.refresh = SimpleNamespace(match=lambda d, hz: seen.append(("match", d, hz)) or "",
                                   restore=lambda: seen.append(("restore",)) or "")
    sink.overlay_of, sink.win = None, object()
    sink.glfw = SimpleNamespace(get_win32_window=lambda win: 1)
    monkeypatch.setattr("framegen.io.sys.platform", "win32")
    import framegen.window
    monkeypatch.setattr(framegen.window, "monitor_of", lambda hwnd: (None, "D", 144))
    sink.settings = SimpleNamespace(fullscreen=True, match_hz=True, hz=60)
    sink._match_refresh()
    sink.settings = SimpleNamespace(fullscreen=False, match_hz=True, hz=60)
    sink._match_refresh()
    sink.settings = SimpleNamespace(fullscreen=True, match_hz=False, hz=60)
    sink._match_refresh()
    sink.overlay_of, sink.settings = object(), SimpleNamespace(fullscreen=True, match_hz=True, hz=60)
    sink._match_refresh()
    assert seen == [("match", "D", 60), ("restore",), ("restore",), ("restore",)]


def test_leaves_the_shared_user32_that_window_py_uses_alone():
    # Live session: after the first switch, window.monitor_of failed with "expected LP__DEVMODE instance instead of
    # pointer to _DEVMODE", so every later fullscreen or output change left the rate as it was
    import ctypes
    import sys

    import pytest
    if sys.platform != "win32":
        pytest.skip("Windows only")
    from framegen import window
    window._api()
    before = ctypes.windll.user32.EnumDisplaySettingsW.argtypes
    hwnd = ctypes.windll.user32.GetDesktopWindow()
    refresh.rates(window.monitor_of(hwnd)[1])
    assert ctypes.windll.user32.EnumDisplaySettingsW.argtypes == before
    window.monitor_of(hwnd)  # still works
