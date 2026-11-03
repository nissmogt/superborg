"""Best-effort probes of the desktop: the window in front, and a screenshot on request.

Both use tools the operating system already ships, so the recorder has no dependencies. Each
returns None (or False) when the tool is missing or refuses, and the recorder carries on without
that detail. Nothing here runs unless a task is being recorded.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from typing import Optional, Tuple

TIMEOUT_S = 3.0

_MAC_FRONT = (
    'tell application "System Events"\n'
    '  set p to first application process whose frontmost is true\n'
    '  set appName to name of p\n'
    '  set winName to ""\n'
    '  try\n'
    '    set winName to name of front window of p\n'
    '  end try\n'
    '  return appName & "\\n" & winName\n'
    'end tell'
)

_WIN_FRONT = r"""
Add-Type @"
using System; using System.Runtime.InteropServices; using System.Text;
public class FG {
  [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
  [DllImport("user32.dll")] public static extern int GetWindowText(IntPtr h, StringBuilder s, int n);
  [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr h, out uint pid);
}
"@
$h = [FG]::GetForegroundWindow()
$sb = New-Object System.Text.StringBuilder 512
[void][FG]::GetWindowText($h, $sb, 512)
$pid0 = 0
[void][FG]::GetWindowThreadProcessId($h, [ref]$pid0)
$name = ""
try { $name = (Get-Process -Id $pid0 -ErrorAction Stop).ProcessName } catch {}
Write-Output $name
Write-Output $sb.ToString()
"""

_WIN_SHOT = r"""
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
$b = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
$bmp = New-Object System.Drawing.Bitmap $b.Width, $b.Height
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.CopyFromScreen($b.Location, [System.Drawing.Point]::Empty, $b.Size)
$bmp.Save($args[0], [System.Drawing.Imaging.ImageFormat]::Png)
"""


def _run(cmd, **kw) -> Optional[str]:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_S, **kw)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout


def foreground_window() -> Optional[Tuple[str, str]]:
    """(application name, window title) of the window in front, or None when unknown."""
    if sys.platform == "darwin":
        out = _run(["osascript", "-e", _MAC_FRONT])
    elif sys.platform.startswith("win"):
        out = _run(["powershell", "-NoProfile", "-NonInteractive", "-Command", _WIN_FRONT])
    else:
        if not shutil.which("xdotool") or not os.environ.get("DISPLAY"):
            return None
        wid = _run(["xdotool", "getactivewindow"])
        if not wid:
            return None
        wid = wid.strip()
        title = _run(["xdotool", "getwindowname", wid]) or ""
        pid = (_run(["xdotool", "getwindowpid", wid]) or "").strip()
        app = ""
        if pid.isdigit():
            try:
                with open(f"/proc/{pid}/comm", encoding="utf-8", errors="replace") as f:
                    app = f.read().strip()
            except OSError:
                pass
        return app, title.strip()
    if out is None:
        return None
    lines = out.rstrip("\n").split("\n")
    app = lines[0].strip() if lines else ""
    title = lines[1].strip() if len(lines) > 1 else ""
    if not app and not title:
        return None
    return app, title


def screenshot(path: str) -> bool:
    """Write a PNG of the main screen to `path`. False when no screenshot tool worked."""
    if sys.platform == "darwin":
        return _run(["screencapture", "-x", path]) is not None and os.path.isfile(path)
    if sys.platform.startswith("win"):
        _run(["powershell", "-NoProfile", "-NonInteractive", "-Command", _WIN_SHOT, path])
        return os.path.isfile(path)
    for cmd in (["gnome-screenshot", "-f", path], ["grim", path], ["scrot", path],
                ["import", "-window", "root", path], ["spectacle", "-b", "-n", "-o", path]):
        if shutil.which(cmd[0]) and _run(cmd) is not None and os.path.isfile(path):
            return True
    return False
