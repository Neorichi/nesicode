"""What the Linux bundle leaves to the system (used by packaging/nesicode.spec; tested by tests/test_v049_linux.py).

The window is WebKitGTK, which cannot travel inside a bundle: it starts helper programs of its own (WebKitWebProcess,
WebKitNetworkProcess) from the system's folders and loads GStreamer plugins from there. So the GTK stack comes from the
system (the .deb depends on it) and so must every library those load: a copy of GLib, libstdc++ or zlib inside the
bundle would be found first and break the system's WebKit with an older version. Only Python itself and the Python
packages' own files go in the bundle; PyGObject's compiled module there links to the system's GObject Introspection.
Build with uv's own Python (as packaging/linux/build_deb.sh does), not the distribution's: its files live under /usr.
"""

from __future__ import annotations

import os

SYSTEM_ROOTS = ("/usr/", "/lib/", "/lib32/", "/lib64/", "/libx32/", "/bin/", "/sbin/", "/etc/")

# PyInstaller's GTK runtime hooks point GTK at copies inside the bundle (typelibs, GIO modules, pixbuf loaders,
# themes, GStreamer plugins). With the system's GTK they would only hide the system's own, and the variables they set
# would reach every program the app starts.
GTK_RUNTIME_HOOKS = frozenset({"pyi_rth_gi", "pyi_rth_gio", "pyi_rth_glib", "pyi_rth_gtk", "pyi_rth_gdkpixbuf",
                               "pyi_rth_gstreamer"})

# Must never be in the bundle (checked after the build): the system's GTK stack and the libraries it shares with it,
# by their system names (a wheel's own renamed copy, libffi-1a2b3c.so.8, is private to that wheel and harmless).
FORBIDDEN = ("libgtk-3.so", "libgdk-3.so", "libwebkit2gtk-4", "libjavascriptcoregtk-4", "libsoup-3.0.so",
             "libsoup-2.4.so", "libglib-2.0.so", "libgobject-2.0.so", "libgio-2.0.so", "libgmodule-2.0.so",
             "libgirepository-1.0.so", "libcairo.so", "libcairo-gobject.so", "libpango-1.0.so", "libpangocairo-1.0.so",
             "libharfbuzz.so", "libfontconfig.so", "libfreetype.so", "libX11.so", "libxcb.so", "libwayland-client.so",
             "libstdc++.so", "libgcc_s.so", "libz.so", "libffi.so", "libexpat.so", "libpcre2-8.so")


def from_system(src: str) -> bool:
    """Is this file one of the system's (under /usr, /lib…), rather than Python's or a package's?"""
    path = os.path.abspath(str(src))
    return path.startswith(SYSTEM_ROOTS) or os.path.realpath(path).startswith(SYSTEM_ROOTS)


def keep_binary(entry) -> bool:
    dest, src = entry[0], entry[1]
    if os.path.basename(str(dest)).startswith("libpython"):
        return True  # the interpreter itself (from wherever this Python is installed)
    return not from_system(src)


def keep_data(entry) -> bool:
    return not from_system(entry[1])


def keep_script(entry) -> bool:
    return entry[0] not in GTK_RUNTIME_HOOKS


def forbidden(names) -> list[str]:
    """The files of a built bundle (relative names) that must have been left to the system."""
    return sorted(n for n in names if os.path.basename(n).startswith(FORBIDDEN))
