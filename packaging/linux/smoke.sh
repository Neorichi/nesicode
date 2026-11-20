#!/usr/bin/env bash
# Installs a Nesicode .deb on this machine (a fresh container or a CI runner) and checks the installed app: every
# library resolves, the menu entry is valid, the app starts (Xvfb + a window manager), its window draws the app, its
# server answers on the Chrome extension's port, nothing it starts inherits the bundle's library path, closing the
# window quits it, and a quick reopen gets the same port back.
# Usage: smoke.sh <package.deb> [out-dir]  — CI only: it installs packages and starts X servers.
set -euo pipefail
DEB=$(readlink -f "$1")
OUT=${2:-smoke-out}; mkdir -p "$OUT"; OUT=$(readlink -f "$OUT")
SUDO=; [ "$(id -u)" = 0 ] || SUDO=sudo
export DEBIAN_FRONTEND=noninteractive
log() { echo "==> $*"; }
fail() { echo "FAIL: $*"; [ -f "${APPLOG:-/nonexistent}" ] && { echo "---- app log ----"; cat "$APPLOG"; }; exit 1; }

log "install $(basename "$DEB") (its dependencies only, no recommended packages)"
$SUDO apt-get update -qq
$SUDO apt-get install -y -qq --no-install-recommends "$DEB" >/dev/null
$SUDO apt-get install -y -qq --no-install-recommends xvfb x11-utils imagemagick openbox wmctrl dbus dbus-x11 curl \
  procps iproute2 desktop-file-utils >/dev/null
grep -h PRETTY_NAME /etc/os-release
dpkg-query -W -f='${Package} ${Version}\n' nesicode libwebkit2gtk-4.1-0 libgtk-3-0 libgtk-3-0t64 2>/dev/null || true

log "libraries"
missing=""
while IFS= read -r -d '' f; do
  nf=$(ldd "$f" 2>/dev/null | grep "not found" || true)
  [ -z "$nf" ] || missing+="$f:"$'\n'"$nf"$'\n'
done < <(find /opt/nesicode -type f \( -name '*.so' -o -name '*.so.*' -o -path /opt/nesicode/nesicode \) -print0)
[ -z "$missing" ] || { echo "$missing"; fail "libraries not found"; }

log "menu entry, command and icons"
desktop-file-validate /usr/share/applications/nesicode.desktop
[ "$(readlink /usr/bin/nesicode)" = /opt/nesicode/nesicode ] || fail "/usr/bin/nesicode"
for s in 16 24 32 48 64 128 256 512; do [ -f "/usr/share/icons/hicolor/${s}x${s}/apps/nesicode.png" ] || fail "icon $s"; done

log "display"
HOME=$(mktemp -d); XDG_RUNTIME_DIR=$(mktemp -d); chmod 700 "$XDG_RUNTIME_DIR"; export HOME XDG_RUNTIME_DIR
APPLOG="$HOME/.local/share/nesicode/nesicode.log"
Xvfb :99 -screen 0 1600x1000x24 -nolisten tcp >"$OUT/xvfb.log" 2>&1 &
export DISPLAY=:99
for _ in $(seq 50); do xdpyinfo >/dev/null 2>&1 && break; sleep 0.2; done
openbox >"$OUT/openbox.log" 2>&1 &
sleep 1

app_pid() { pgrep -x nesicode | head -1; }
start_app() { (dbus-run-session -- nesicode >>"$OUT/app.out" 2>&1 &); }
wait_port() {
  for _ in $(seq 180); do
    code=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:47821/api/health || true)
    case "$code" in 200|403) return 0 ;; esac
    sleep 0.5
  done
  return 1
}
close_app() {
  local pid=$1
  wmctrl -c Nesicode || true
  for _ in $(seq 60); do kill -0 "$pid" 2>/dev/null || return 0; sleep 0.5; done
  return 1
}

log "start"
start_app
wait_port || fail "no server on 127.0.0.1:47821"
APP=$(app_pid); [ -n "$APP" ] || fail "no nesicode process"
ss -ltnp | grep ':47821' || true
for _ in $(seq 120); do xwininfo -root -tree | grep -q '"Nesicode"' && break; sleep 0.5; done
xwininfo -root -tree | grep '"Nesicode"' || fail "no Nesicode window"
xprop -name Nesicode WM_CLASS | tee "$OUT/wm_class.txt"
grep -q '"Nesicode"' "$OUT/wm_class.txt" || fail "the window's class does not match the menu entry (StartupWMClass)"

log "the window draws the app"
colors=0
for _ in $(seq 30); do
  sleep 2
  import -window root "$OUT/screen.png" 2>/dev/null || magick import -window root "$OUT/screen.png"
  colors=$(identify -format '%k' "$OUT/screen.png" 2>/dev/null || magick identify -format '%k' "$OUT/screen.png")
  [ "$colors" -gt 200 ] && break
done
echo "screenshot colors: $colors"
[ "$colors" -gt 200 ] || fail "the window looks blank ($colors colours)"

log "what the app starts does not inherit the bundle's libraries"
echo "app $APP: $(tr '\0' '\n' </proc/"$APP"/environ | grep -E '^LD_LIBRARY_PATH' || echo 'no LD_LIBRARY_PATH at exec')"
descendants() { for c in $(pgrep -P "$1"); do echo "$c"; descendants "$c"; done; }
found_child=0
for c in $(descendants "$APP"); do
  [ -r "/proc/$c/environ" ] || continue
  found_child=1
  name=$(cat "/proc/$c/comm" 2>/dev/null || echo "?")
  ldp=$(tr '\0' '\n' <"/proc/$c/environ" | grep -E '^(LD_LIBRARY_PATH|GI_TYPELIB_PATH|GTK_PATH|GIO_MODULE_DIR|GDK_PIXBUF_MODULE_FILE)=' || true)
  echo "  $c $name: ${ldp:-clean}"
  case "$ldp" in *"/opt/nesicode"*) fail "$name inherited the bundle's paths" ;; esac
done
[ "$found_child" = 1 ] || fail "no child processes (WebKit's) found"

log "app log"
cat "$APPLOG"
if grep -E "Traceback|could not|setup step failed|check failed|cannot be loaded|not be secured" "$APPLOG"; then
  fail "errors in the app log"
fi

log "closing the window quits the app"
close_app "$APP" || fail "the app is still running after its window was closed"

log "a quick reopen keeps port 47821"
ss -tan | grep -c ':47821 .*TIME-WAIT' || true
start_app
wait_port || fail "the reopened app is not on 127.0.0.1:47821"
APP2=$(app_pid)
ss -ltnp | grep ':47821' | grep -q "pid=$APP2," || fail "47821 is not the reopened app's"
for _ in $(seq 120); do xwininfo -root -tree | grep -q '"Nesicode"' && break; sleep 0.5; done
close_app "$APP2" || fail "the reopened app did not quit"
echo "OK: $(basename "$DEB")"
