// App window mode's shortcuts. framegen/hotkeys.py does the same in the pipeline; keep the two in step. SmoovPlay's
// picture sits over the app without taking the keyboard (some apps stop drawing without it), so SmoovPlay's own keys
// come through Windows' system-wide shortcuts: Ctrl+Alt with the usual key by default. People can change them on
// the Play page; they're kept in the launcher's prefs as "window-key:<action>=<shortcut>" and passed to
// run.py --window-keys. The pipeline says which it got ("[window-keys] {json}"), shown on the Play page.
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Web.Script.Serialization;
using System.Windows.Input;

namespace SmoovPlay
{
    static class WindowKeys
    {
        public class Act
        {
            public string Id, Label;
            public string[] Defaults;  // the first free one wins
        }

        public static readonly Act[] Actions =
        {
            new Act { Id = "menu", Label = "Menu", Defaults = new[] { "Ctrl+Alt+M", "Ctrl+Alt+Space" } },
            new Act { Id = "fullscreen", Label = "Fullscreen", Defaults = new[] { "Ctrl+Alt+F" } },
            new Act { Id = "quit", Label = "Quit", Defaults = new[] { "Ctrl+Alt+Esc", "Ctrl+Alt+End" } },
            new Act { Id = "stats", Label = "Stats overlay", Defaults = new[] { "Ctrl+Alt+S" } },
            new Act { Id = "smooth", Label = "Smooth on/off", Defaults = new[] { "Ctrl+Alt+I" } },
        };

        // Windows' own, or keys people need to get out of things
        static readonly HashSet<string> Reserved = new HashSet<string> { "Ctrl+Alt+Delete", "Ctrl+Shift+Esc", "Alt+Tab", "Alt+Esc", "Alt+F4", "Ctrl+Alt+Tab", "Win+L", "Win+D", "Win+Tab", "Ctrl+Esc" };

        // names as framegen/hotkeys.py writes them, by Windows' virtual-key code
        static readonly Dictionary<int, string> Names = BuildNames();

        static Dictionary<int, string> BuildNames()
        {
            var d = new Dictionary<int, string>
            {
                { 0x1B, "Esc" }, { 0x20, "Space" }, { 0x0D, "Enter" }, { 0x08, "Backspace" }, { 0x09, "Tab" }, { 0x23, "End" },
                { 0x24, "Home" }, { 0x21, "PageUp" }, { 0x22, "PageDown" }, { 0x2D, "Insert" }, { 0x2E, "Delete" }, { 0x13, "Pause" },
                { 0x25, "Left" }, { 0x26, "Up" }, { 0x27, "Right" }, { 0x28, "Down" },
            };
            for (int c = 'A'; c <= 'Z'; c++) d[c] = ((char)c).ToString();
            for (int c = '0'; c <= '9'; c++) d[c] = ((char)c).ToString();
            for (int n = 1; n <= 24; n++) d[0x6F + n] = "F" + n;
            return d;
        }

        const string Prefix = "window-key:";
        const int ModAlt = 0x1, ModCtrl = 0x2, ModShift = 0x4, ModWin = 0x8, NoRepeat = 0x4000;

        // From the last session: what each action got (null: nothing free), and the shortcuts another program had
        public static Dictionary<string, string> Live;
        public static List<string> LiveTaken = new List<string>();

        public static Act Find(string id) { return Actions.First(a => a.Id == id); }

        public static Dictionary<string, string> Picked(HashSet<string> prefs)
        {
            var d = new Dictionary<string, string>();
            foreach (var f in prefs.Where(x => x.StartsWith(Prefix)))
            {
                var kv = f.Substring(Prefix.Length).Split(new[] { '=' }, 2);
                if (kv.Length == 2 && Actions.Any(a => a.Id == kv[0]) && kv[1].Length > 0) d[kv[0]] = kv[1];
            }
            return d;
        }

        /// <summary>combo null: back to the default.</summary>
        public static void Pick(HashSet<string> prefs, string action, string combo)
        {
            prefs.RemoveWhere(x => x.StartsWith(Prefix + action + "="));
            if (combo != null) prefs.Add(Prefix + action + "=" + combo);
        }

        public static void ResetAll(HashSet<string> prefs) { prefs.RemoveWhere(x => x.StartsWith(Prefix)); }

        /// <summary>run.py --window-keys: "menu=Ctrl+Alt+K;quit=Ctrl+Alt+End", or null when all are the defaults.</summary>
        public static string Arg(HashSet<string> prefs)
        {
            var p = Picked(prefs);
            return p.Count == 0 ? null : string.Join(";", p.Select(kv => kv.Key + "=" + kv.Value));
        }

        /// <summary>Each action's shortcuts to try, as framegen/hotkeys.choices: the one picked, or its defaults
        /// minus any picked for another action.</summary>
        static Dictionary<string, string[]> Choices(HashSet<string> prefs)
        {
            var p = Picked(prefs);
            var used = new HashSet<string>(p.Values);
            return Actions.ToDictionary(a => a.Id, a => p.ContainsKey(a.Id) ? new[] { p[a.Id] } : a.Defaults.Where(c => !used.Contains(c)).ToArray());
        }

        /// <summary>What each action gets on this PC now (as the pipeline will: the first free one), and the
        /// shortcuts another program has. Only while SmoovPlay isn't playing: its own would look taken.</summary>
        public static Dictionary<string, string> Resolve(HashSet<string> prefs, List<string> taken)
        {
            var got = new Dictionary<string, string>();
            foreach (var kv in Choices(prefs))
            {
                got[kv.Key] = null;
                foreach (var c in kv.Value)
                {
                    if (Free(c)) { got[kv.Key] = c; break; }
                    if (taken != null) taken.Add(c);
                }
            }
            return got;
        }

        /// <summary>While playing: what the pipeline said it got, or (before it says) what it was asked for.</summary>
        public static Dictionary<string, string> Playing(HashSet<string> prefs)
        {
            return Live ?? Choices(prefs).ToDictionary(kv => kv.Key, kv => kv.Value.FirstOrDefault());
        }

        public static void FromSession(string json)
        {
            try
            {
                var root = (Dictionary<string, object>)new JavaScriptSerializer().DeserializeObject(json);
                var keys = (Dictionary<string, object>)root["keys"];
                Live = keys.ToDictionary(kv => kv.Key, kv => kv.Value as string);
                LiveTaken = ((object[])root["taken"]).Select(o => o as string).Where(s => s != null).ToList();
            }
            catch (Exception) { }
        }

        /// <summary>"Ctrl+Alt+M menu, Ctrl+Alt+F fullscreen, Ctrl+Alt+Esc quit": for the Play page's lines.</summary>
        public static string Sentence(Dictionary<string, string> got)
        {
            var parts = new[] { "menu", "fullscreen", "quit" }.Where(a => got.ContainsKey(a) && got[a] != null)
                .Select(a => got[a] + " " + (a == "menu" ? "opens the menu" : a == "fullscreen" ? "goes fullscreen" : "quits")).ToList();
            if (parts.Count == 0) return "";
            return parts.Count == 1 ? parts[0] : string.Join(", ", parts.Take(parts.Count - 1)) + " and " + parts.Last();
        }

        // ---- reading and checking a shortcut ------------------------------------------------------------------
        /// <summary>A key press in the editor, as a shortcut's name ("Ctrl+Alt+K"); null while only Ctrl, Alt, Shift
        /// or Win are down; "?" for a key that can't be used.</summary>
        public static string FromKey(Key key, ModifierKeys mods)
        {
            switch (key)
            {
                case Key.LeftCtrl: case Key.RightCtrl: case Key.LeftAlt: case Key.RightAlt: case Key.LeftShift:
                case Key.RightShift: case Key.LWin: case Key.RWin: case Key.System: case Key.None: case Key.ImeProcessed:
                case Key.DeadCharProcessed:
                    return null;
            }
            string name;
            if (!Names.TryGetValue(KeyInterop.VirtualKeyFromKey(key), out name)) return "?";
            var parts = new List<string>();
            if ((mods & ModifierKeys.Control) != 0) parts.Add("Ctrl");
            if ((mods & ModifierKeys.Alt) != 0) parts.Add("Alt");
            if ((mods & ModifierKeys.Shift) != 0) parts.Add("Shift");
            if ((mods & ModifierKeys.Windows) != 0) parts.Add("Win");
            parts.Add(name);
            return string.Join("+", parts);
        }

        static bool Parse(string name, out int mods, out int vk)
        {
            mods = 0; vk = 0;
            if (string.IsNullOrEmpty(name)) return false;
            var parts = name.Split('+');
            foreach (var p in parts.Take(parts.Length - 1))
            {
                if (p == "Ctrl") mods |= ModCtrl;
                else if (p == "Alt") mods |= ModAlt;
                else if (p == "Shift") mods |= ModShift;
                else if (p == "Win") mods |= ModWin;
                else return false;
            }
            var key = Names.FirstOrDefault(kv => kv.Value == parts.Last());
            vk = key.Key;
            return key.Value != null;
        }

        /// <summary>Why `name` can't be the shortcut for `action`, or null. Not while SmoovPlay is playing (its own
        /// shortcuts would look taken).</summary>
        public static string Problem(string name, string action, HashSet<string> prefs)
        {
            int mods, vk;
            if (name == "?" || !Parse(name, out mods, out vk))
                return "That key can't be used. Pick a letter, a number, F1 to F24, Esc, Space, Enter, Tab, an arrow, Home, End, Page Up, Page Down, Insert, Delete or Pause.";
            if ((mods & (ModCtrl | ModAlt | ModWin)) == 0)
                return "Add Ctrl, Alt or Win: on its own (or with Shift only), the key would stop working in the game and the app.";
            if (Reserved.Contains(name)) return name + " belongs to Windows.";
            var others = Resolve(prefs, null).Where(kv => kv.Key != action && kv.Value == name).Select(kv => Find(kv.Key).Label).FirstOrDefault();
            if (others != null) return others + " already uses " + name + ". Change that one first.";
            if (!Free(name)) return "Another program on this PC already uses " + name + ". Pick another.";
            return null;
        }

        [DllImport("user32.dll")] static extern bool RegisterHotKey(IntPtr hwnd, int id, int mods, int vk);
        [DllImport("user32.dll")] static extern bool UnregisterHotKey(IntPtr hwnd, int id);

        /// <summary>No other program has it: it's registered for a moment and let go straight away.</summary>
        public static bool Free(string name)
        {
            int mods, vk;
            if (!Parse(name, out mods, out vk)) return false;
            if (!RegisterHotKey(IntPtr.Zero, 0xBF00, mods | NoRepeat, vk)) return false;
            UnregisterHotKey(IntPtr.Zero, 0xBF00);
            return true;
        }
    }
}
