"""App window mode's shortcuts (framegen/hotkeys.py): reading and checking them, falling back when another program has
one, the menu's keys while it's open, and the video window acting on them as on its own keys. Windows' calls are
stand-ins: nothing is registered for real."""
import ctypes
from types import SimpleNamespace

import pytest

from framegen import hotkeys, io
from framegen.hotkeys import Hotkeys, describe, parse, parse_arg

CTRL_ALT = 0x0002 | 0x0001


class _User32:
    """RegisterHotKey and friends, with some shortcuts taken by "another program"."""

    def __init__(self, taken=()):
        self.taken = {(parse(t).mods, parse(t).vk) for t in taken}
        self.on: dict[int, tuple[int, int]] = {}
        self.queue: list = []  # a hotkey's id, or (message, hwnd) for anything else
        self.filters: list[tuple[int, int]] = []  # each PeekMessageW's (lowest, highest) message
        self.dispatched: list[int] = []
        self.quit: list[int] = []

    def RegisterHotKey(self, hwnd, i, mods, vk):
        key = (mods & ~hotkeys.NOREPEAT, vk)
        if key in self.taken or i in self.on or key in self.on.values():
            return 0
        self.on[i] = key
        return 1

    def UnregisterHotKey(self, hwnd, i):
        return 1 if self.on.pop(i, None) else 0

    def PeekMessageW(self, msg, hwnd, lo, hi, remove):
        self.filters.append((lo, hi))
        if not self.queue:
            return 0
        item = self.queue.pop(0)
        if isinstance(item, tuple):
            msg._obj.message, msg._obj.hWnd, msg._obj.wParam = item[0], item[1], 0
        else:
            msg._obj.message, msg._obj.hWnd, msg._obj.wParam = hotkeys.WM_HOTKEY, None, item
        return 1

    def TranslateMessage(self, msg):
        return 0

    def DispatchMessageW(self, msg):
        self.dispatched.append(msg._obj.message)
        return 0

    def PostQuitMessage(self, code):
        self.quit.append(code)

    def press(self, name):
        c = parse(name) if "+" in name else SimpleNamespace(mods=0, vk=hotkeys.KEYS[name])
        self.queue += [i for i, k in self.on.items() if k == (c.mods, c.vk)]


def test_names_are_read_the_way_people_write_them():
    assert parse("ctrl + alt + escape").name == "Ctrl+Alt+Esc"
    assert parse("Alt+Ctrl+m") == hotkeys.Combo(CTRL_ALT, ord("M"), "Ctrl+Alt+M")
    assert parse("Ctrl+Shift+F5").name == "Ctrl+Shift+F5"
    assert parse("Win+Alt+PgDn").name == "Alt+Win+PageDown"


@pytest.mark.parametrize("text, why", [
    ("M", "needs Ctrl, Alt or Win"), ("Shift+M", "needs Ctrl, Alt or Win"), ("Ctrl+Alt", "needs a key"),
    ("Ctrl+Alt+M+K", "more than one key"), ("Ctrl+Alt+Delete", "belongs to Windows"), ("Alt+Tab", "belongs to Windows"),
    ("Ctrl+Alt+Banana", "unknown key"), ("Ctrl+Ctrl+M", "Ctrl twice"),
])
def test_shortcuts_that_cant_be_used_say_why(text, why):
    with pytest.raises(ValueError, match=why):
        parse(text)


def test_picked_shortcuts_are_checked():
    assert parse_arg("menu=ctrl+alt+k; quit=Ctrl+Alt+End;") == {"menu": "Ctrl+Alt+K", "quit": "Ctrl+Alt+End"}
    assert parse_arg(None) == {} and parse_arg("") == {}
    with pytest.raises(ValueError, match="no action 'jump'"):
        parse_arg("jump=Ctrl+Alt+J")
    with pytest.raises(ValueError, match="same shortcut"):
        parse_arg("menu=Ctrl+Alt+K;quit=Alt+Ctrl+K")
    assert parse_arg("menu=Ctrl+Alt+R") == {"menu": "Ctrl+Alt+R"}  # (once instant replay's, now free)
    with pytest.raises(ValueError, match="menu: 'M'"):
        parse_arg("menu=M")


def test_defaults_are_ctrl_alt_and_the_usual_key():
    u = _User32()
    h = Hotkeys(user32=u)
    assert h.got == {"menu": "Ctrl+Alt+M", "quit": "Ctrl+Alt+Esc", "fullscreen": "Ctrl+Alt+F",
                     "stats": "Ctrl+Alt+S", "smooth": "Ctrl+Alt+I"}
    assert h.taken == [] and len(u.on) == 5
    assert describe(h.got).split("\n")[:3] == ["Ctrl+Alt+M: menu", "Ctrl+Alt+F: fullscreen", "Ctrl+Alt+Esc: quit"]
    assert describe(h.got).isascii()
    h.close()
    assert u.on == {}


def test_a_shortcut_another_program_has_falls_back_and_says_so():
    h = Hotkeys(user32=_User32(taken=["Ctrl+Alt+M", "Ctrl+Alt+S"]))
    assert h.got["menu"] == "Ctrl+Alt+Space" and h.got["stats"] is None
    assert h.taken == ["Ctrl+Alt+M", "Ctrl+Alt+S"]
    assert "stats" not in describe(h.got)
    assert '"menu": "Ctrl+Alt+Space"' in h.report() and h.report().startswith("[window-keys] {")


def test_a_picked_shortcut_replaces_the_default_and_isnt_used_twice():
    h = Hotkeys({"quit": "Ctrl+Alt+M"}, user32=_User32())
    assert h.got["quit"] == "Ctrl+Alt+M"
    assert h.got["menu"] == "Ctrl+Alt+Space"  # its default went to quit


def test_the_menus_keys_are_taken_only_while_it_is_open():
    u = _User32()
    h = Hotkeys(user32=u)
    u.press("Down")
    assert h.poll() == []  # the app keeps its arrows
    h.set_nav(True)
    u.press("Down"), u.press("Ctrl+Alt+M")
    assert h.poll() == ["nav:Down", "menu"]
    h.set_nav(False)
    assert len(u.on) == 5


def _sink(u):
    """A video window over an app, without GL or a window: just what acting on the keys needs."""
    keys = "M TAB ESCAPE Q I S U Z F L UP DOWN LEFT RIGHT ENTER KP_ENTER BACKSPACE".split()
    glfw = SimpleNamespace(PRESS=1, REPEAT=2, MOD_SHIFT=1, poll_events=lambda: None,
                           **{"KEY_" + k: 1000 + i for i, k in enumerate(keys)})
    menu = SimpleNamespace(open=False, pressed=[])
    menu.key = lambda k: menu.pressed.append(k) or ("close" if k == "back" else None)  # (Esc on the main page)
    st = SimpleNamespace(interp=True, overlay=False, overlay_style="minimal", fullscreen=False, changed=[])
    st.override = lambda k, v: st.changed.append((k, v))
    st.set = lambda k, v: (st.changed.append((k, v)), setattr(st, k, v))
    w = io.GLWindowSink.__new__(io.GLWindowSink)
    w.glfw, w.win, w.menu, w.settings, w.cmd, w._toast_until = glfw, object(), menu, st, None, 0.0
    w.hotkeys = Hotkeys(user32=u)
    return w


def test_the_video_window_acts_on_them_as_on_its_own_keys():
    u = _User32()
    w = _sink(u)
    u.press("Ctrl+Alt+M")
    w._events()
    assert w.menu.open and w.hotkeys.nav_on  # the menu opened, and has its keys
    u.press("Down"), u.press("Enter")
    w._events()
    assert w.menu.pressed == ["down", "enter"]
    u.press("Esc")  # closes the menu (on its main page), and the app has its keys back
    w._events()
    assert w.menu.pressed[-1] == "back" and not w.menu.open and not w.hotkeys.nav_on and w.cmd is None
    u.press("Ctrl+Alt+F"), u.press("Ctrl+Alt+I")
    w._events()
    assert ("fullscreen", True) in w.settings.changed and ("interp", False) in w.settings.changed
    u.press("Ctrl+Alt+Esc")
    w._events()
    assert w.cmd == "quit"


def test_a_message_for_another_id_is_ignored():
    u = _User32()
    h = Hotkeys(user32=u)
    u.queue.append(42)
    assert h.poll() == []


def test_poll_takes_every_message_and_hands_the_rest_to_their_windows():
    """Looking for WM_HOTKEY alone (a filtered PeekMessage) made Windows call the video window hung after 5 s
    ("Python is not responding"). Every message is taken; the window's own go to it, as GLFW would send them."""
    u = _User32()
    h = Hotkeys(user32=u)
    u.queue += [(0x0200, 77), 1, (0x000F, 77)]  # a mouse move, Ctrl+Alt+M, a paint
    assert h.poll() == ["menu"]
    assert u.dispatched == [0x0200, 0x000F]
    assert set(u.filters) == {(0, 0)}


def test_poll_leaves_quit_to_glfw():
    u = _User32()
    h = Hotkeys(user32=u)
    u.queue += [(hotkeys.WM_QUIT, None), (0x000F, 77)]
    assert h.poll() == []
    assert u.quit == [0] and u.dispatched == []  # put back for GLFW, which reads the rest


def test_msg_stand_in_works_with_byref():
    from ctypes import wintypes as wt
    msg = wt.MSG()
    u = _User32()
    u.queue.append(3)
    assert u.PeekMessageW(ctypes.byref(msg), None, 0, 0, 1) and msg.wParam == 3
