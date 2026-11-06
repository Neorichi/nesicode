#!/usr/bin/env bash
# Builds dist/Nesicode.app and dist/nesicode-<version>-macos-<arch>.dmg
set -euo pipefail
cd "$(dirname "$0")/.."
VERSION=$(grep -m1 '^version' pyproject.toml | cut -d'"' -f2)
ARCH=$(uname -m)

echo "==> web"
(cd web && npm ci --silent && npm run build --silent)

echo "==> python deps"
uv sync --extra desktop --group build

echo "==> app bundle"
rm -rf build dist/nesicode dist/Nesicode.app
uv run pyinstaller packaging/nesicode.spec --noconfirm --log-level WARN

echo "==> ad-hoc signature (no Apple Developer ID; Gatekeeper will ask the first time)"
codesign --force --deep --sign - dist/Nesicode.app

echo "==> dmg"
STAGE=$(mktemp -d)
cp -R dist/Nesicode.app "$STAGE/"
ln -s /Applications "$STAGE/Applications"
DMG="dist/Nesicode-${VERSION}-macos-${ARCH}.dmg"
rm -f "$DMG"
hdiutil create -volname "Nesicode" -srcfolder "$STAGE" -ov -format UDZO "$DMG" >/dev/null
rm -rf "$STAGE"
echo "OK: dist/Nesicode.app and $DMG"
