#!/usr/bin/env bash
# Builds dist/nesicode/ (PyInstaller) and dist/Nesicode-<version>-linux-<arch>.deb on Ubuntu/Debian.
# Needs: uv, Node 20+, and the headers PyGObject builds against plus the GTK stack the window uses:
#   sudo apt-get install -y pkg-config libgirepository1.0-dev libcairo2-dev gir1.2-gtk-3.0 gir1.2-webkit2-4.1
# Build on the oldest release to support (Ubuntu 22.04 for 22.04+ and Debian 12+): the bundle needs that glibc or newer.
set -euo pipefail
cd "$(dirname "$0")/../.."
VERSION=$(grep -m1 '^version' pyproject.toml | cut -d'"' -f2)

for pc in gobject-introspection-1.0 cairo; do
  pkg-config --exists "$pc" || { echo "missing $pc headers: sudo apt-get install -y pkg-config libgirepository1.0-dev libcairo2-dev"; exit 1; }
done

echo "==> web"
(cd web && npm ci --silent && npm run build --silent)

echo "==> python deps (uv's own Python: the bundle must not take files from /usr, see packaging/linux/bundle.py)"
uv sync --extra desktop --group build --python-preference only-managed

echo "==> bundle"
rm -rf build/nesicode dist/nesicode
uv run --no-sync pyinstaller packaging/nesicode.spec --noconfirm --log-level WARN

echo "==> checks"
uv run --no-sync python - <<'EOF'
import os, sys
sys.path.insert(0, "packaging/linux")
import bundle
names = [os.path.relpath(os.path.join(d, f), "dist/nesicode") for d, _, fs in os.walk("dist/nesicode") for f in fs]
bad = bundle.forbidden(names)
assert not bad, f"system libraries inside the bundle: {bad}"
need = ["_internal/web_dist/index.html", "_internal/chrome_extension/manifest.json", "_internal/nesicode.png"]
missing = [n for n in need if n not in names]
assert not missing, f"missing from the bundle: {missing}"
assert any(os.path.basename(n).startswith("_gi.") for n in names), "PyGObject is not in the bundle"
assert not any(n.startswith("_internal/gi_typelibs/") for n in names), "typelibs inside the bundle"
print(f"bundle ok: {len(names)} files")
EOF

echo "==> deb"
uv run --no-sync python packaging/linux/make_deb.py --version "$VERSION"
