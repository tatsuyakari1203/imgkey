from __future__ import annotations

import subprocess
import sys
from typing import Any


def hidden_subprocess_kwargs(base: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return subprocess kwargs that keep probe/tool children hidden on Windows.

    GUI builds of ImgKey run without a console. Launching console-subsystem tools
    such as nvidia-smi, vswhere, dxc, fxc, dumpbin, or llvm-objdump from that
    process can briefly create a visible console window unless the child is
    started with CREATE_NO_WINDOW. Keep this helper tiny and stdlib-only so CLI
    probes can share the same behavior without changing their stdout/stderr
    capture semantics.
    """

    kwargs = dict(base or {})
    if sys.platform != "win32":
        return kwargs

    kwargs["creationflags"] = int(kwargs.get("creationflags") or 0) | int(getattr(subprocess, "CREATE_NO_WINDOW", 0))
    startupinfo = kwargs.get("startupinfo")
    if startupinfo is None:
        startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= int(getattr(subprocess, "STARTF_USESHOWWINDOW", 0))
    startupinfo.wShowWindow = int(getattr(subprocess, "SW_HIDE", 0))
    kwargs["startupinfo"] = startupinfo
    return kwargs


def run_hidden(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    """Run a subprocess with hidden-window flags on Windows."""

    return subprocess.run(args, **hidden_subprocess_kwargs(kwargs))
