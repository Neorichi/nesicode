# PyInstaller recipe for the nesicode desktop app (macOS .app and Windows folder for the installer).
# Build from the repo root:  uv run pyinstaller packaging/nesicode.spec --noconfirm
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = Path(SPECPATH).parent
VERSION = "0.3.1"
WEB = ROOT / "web" / "dist"
if not (WEB / "index.html").exists():
    raise SystemExit("Build the web first: cd web && npm ci && npm run build")

hidden = (
    collect_submodules("uvicorn")
    + collect_submodules("nesicode")
    + collect_submodules("webview")
    + collect_submodules("docx") + collect_submodules("pptx") + collect_submodules("openpyxl")
    + ["anthropic", "pypdf", "multipart", "python_multipart"]
    + (["Speech", "AVFoundation"] if sys.platform == "darwin" else [])
)
datas = [(str(WEB), "web_dist"), (str(ROOT / "extension"), "chrome_extension")] + collect_data_files("webview") + collect_data_files("anthropic") \
    + collect_data_files("docx") + collect_data_files("pptx") + collect_data_files("openpyxl") \
    + (collect_data_files("tzdata") if sys.platform == "win32" else [])

a = Analysis(
    [str(ROOT / "packaging" / "launcher.py")],
    pathex=[str(ROOT / "src")],
    datas=datas,
    hiddenimports=hidden,
    excludes=["tkinter", "pytest", "PyInstaller"],
    noarchive=False,
)
pyz = PYZ(a.pure)

if sys.platform == "darwin":
    exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="nesicode", console=False,
              icon=str(ROOT / "packaging" / "icons" / "nesicode.icns"))
    coll = COLLECT(exe, a.binaries, a.datas, name="nesicode")
    app = BUNDLE(
        coll,
        name="Nesicode.app",
        icon=str(ROOT / "packaging" / "icons" / "nesicode.icns"),
        bundle_identifier="com.nesilabs.nesicode",
        version=VERSION,
        info_plist={
            "CFBundleName": "Nesicode",
            "CFBundleDisplayName": "Nesicode",
            "CFBundleShortVersionString": VERSION,
            "CFBundleVersion": VERSION,
            "LSMinimumSystemVersion": "12.0",
            "NSHighResolutionCapable": True,
            "LSApplicationCategoryType": "public.app-category.developer-tools",
            "NSMicrophoneUsageDescription": "Nesicode usa el micrófono para que puedas dictar tus mensajes.",
            "NSSpeechRecognitionUsageDescription": "Nesicode transcribe tus mensajes de voz con el reconocimiento de voz de macOS (en el propio Mac cuando es posible).",
        },
    )
else:
    exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="Nesicode", console=False,
              icon=str(ROOT / "packaging" / "icons" / "nesicode.ico"))
    coll = COLLECT(exe, a.binaries, a.datas, name="nesicode")
