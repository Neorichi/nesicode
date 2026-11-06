"""Nesicode icon: white monospace 'n²' on a teal rounded square (same as the web favicon)."""
import sys
from PIL import Image, ImageDraw, ImageFont

TEAL = (14, 143, 126, 255)  # #0e8f7e
FONT = ("/System/Library/Fonts/Menlo.ttc", 1)  # Menlo Bold


def render(size: int, margin: float) -> Image.Image:
    S = size * 4  # supersample for smooth edges
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    m = int(S * margin)
    box = S - 2 * m
    d.rounded_rectangle([m, m, S - m, S - m], radius=int(box * 0.225), fill=TEAL)
    font = ImageFont.truetype(FONT[0], int(box * 0.53), index=FONT[1])
    text = "n²"
    l, t, r, b = d.textbbox((0, 0), text, font=font)
    x = (S - (r - l)) / 2 - l
    y = (S - (b - t)) / 2 - t + box * 0.02
    d.text((x, y), text, font=font, fill="white")
    return img.resize((size, size), Image.LANCZOS)


out = sys.argv[1]
# macOS: the artwork sits inside ~80% of the canvas (Apple icon grid); Windows/extension: nearly full bleed
render(1024, 0.098).save(f"{out}/nesicode.png")
iconset = f"{out}/nesicode.iconset"
import os
os.makedirs(iconset, exist_ok=True)
for s in (16, 32, 128, 256, 512):
    render(s, 0.098).save(f"{iconset}/icon_{s}x{s}.png")
    render(s * 2, 0.098).save(f"{iconset}/icon_{s}x{s}@2x.png")
full = render(256, 0.02)
full.save(f"{out}/nesicode.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
for s in (16, 32, 48, 128):
    render(s, 0.02).save(f"{out}/ext-icon{s}.png")
