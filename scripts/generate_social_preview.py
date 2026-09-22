#!/usr/bin/env python3
"""
Generate the official 1200x630 social preview card for MZSAE.
Produces crisp, high-resolution Open Graph / Twitter summary_large_image card
adhering to the Editorial Research Design System.
"""

from PIL import Image, ImageDraw, ImageFont
import os
import urllib.request

def ensure_font(name: str, url: str) -> str:
    path = f"/tmp/{name}.ttf"
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)
    return path

def generate():
    W, H = 1200, 630
    img = Image.new("RGB", (W, H), color="#FAF9F5")
    draw = ImageDraw.Draw(img)

    # Fetch exact Google Fonts
    serif_path = ensure_font("SourceSerif4", "https://raw.githubusercontent.com/google/fonts/main/ofl/sourceserif4/SourceSerif4%5Bopsz%2Cwght%5D.ttf")
    sans_path = ensure_font("Inter", "https://raw.githubusercontent.com/google/fonts/main/ofl/inter/Inter%5Bopsz%2Cwght%5D.ttf")
    mono_path = ensure_font("JetBrainsMono", "https://raw.githubusercontent.com/google/fonts/main/ofl/jetbrainsmono/JetBrainsMono%5Bwght%5D.ttf")

    font_serif_logo = ImageFont.truetype(serif_path, 42)
    font_serif_title = ImageFont.truetype(serif_path, 52)
    font_sans_eyebrow = ImageFont.truetype(sans_path, 14)
    font_sans_lede = ImageFont.truetype(sans_path, 21)
    font_sans_bold = ImageFont.truetype(sans_path, 16)
    font_sans_small = ImageFont.truetype(sans_path, 13)
    font_mono_val = ImageFont.truetype(mono_path, 44)
    font_mono_pill = ImageFont.truetype(mono_path, 15)

    # Outer subtle hairline frame
    draw.rectangle([20, 20, W - 21, H - 21], outline="#E4E0D6", width=1)

    # Decorative manifold arcs on top-right
    for r in range(120, 480, 40):
        bbox = [W - 250 - r, -r + 100, W - 250 + r, 100 + r]
        draw.arc(bbox, start=60, end=190, fill="#F0EDE6", width=2)
    for r in range(160, 400, 60):
        bbox = [W - 250 - r, -r + 100, W - 250 + r, 100 + r]
        draw.arc(bbox, start=80, end=170, fill="#EBD8D0", width=2)

    # Top Bar: Logo
    draw.text((60, 50), "MZ", fill="#191919", font=font_serif_logo)
    mz_w = draw.textlength("MZ", font=font_serif_logo)
    draw.text((60 + mz_w, 50), "SAE", fill="#C96442", font=font_serif_logo)

    # Top right pill badge
    badge_text = "NEUROMORPHIC SPARSE ATTENTION · V1.3.0"
    bw = draw.textlength(badge_text, font=font_sans_bold)
    draw.rounded_rectangle([W - 60 - bw - 28, 50, W - 60, 92], radius=21, fill="#F0EDE6", outline="#E4E0D6", width=1)
    draw.text((W - 60 - bw - 14, 61), badge_text, fill="#C96442", font=font_sans_bold)

    # Hairline divider
    draw.line([(60, 118), (W - 60, 118)], fill="#E4E0D6", width=1)

    # Eyebrow
    draw.text((60, 142), "EDGE INFERENCE • APPLE SILICON & ACCELERATORS", fill="#8A867D", font=font_sans_eyebrow)

    # Title
    draw.text((60, 172), "Attention that reads only what matters.", fill="#191919", font=font_serif_title)

    # Lede
    lede = (
        "Eliminates the memory wall during autoregressive decoding with 64-byte L2 sentinels\n"
        "and biologically-inspired continual eviction. Bit-exact outputs at 8.4× speedup."
    )
    draw.text((60, 248), lede, fill="#55524B", font=font_sans_lede, spacing=6)

    # 4 Metric Cards
    card_w = 252
    gap = 24
    cards_x = 60
    cards_y = 352
    card_h = 144

    metrics_data = [
        ("8.4×", "#C96442", "Faster Decode", "vs dense SDPA at 11.6k"),
        ("96.7%", "#191919", "DRAM Pruned", "payload traffic in L2"),
        ("10/10", "#191919", "Bit-Exact Retrieval", "dual-panel verified"),
        ("3.28×", "#191919", "KV Compression", "2-bit residual cache"),
    ]

    for i, (val, val_color, label, sub) in enumerate(metrics_data):
        cx = cards_x + i * (card_w + gap)
        draw.rounded_rectangle([cx, cards_y, cx + card_w, cards_y + card_h], radius=10, fill="#FFFFFF", outline="#E4E0D6", width=1)
        draw.text((cx + 20, cards_y + 18), val, fill=val_color, font=font_mono_val)
        draw.text((cx + 20, cards_y + 80), label, fill="#191919", font=font_sans_bold)
        draw.text((cx + 20, cards_y + 106), sub, fill="#8A867D", font=font_sans_small)

    # Bottom Bar: Install pill
    pill_str = "$ pip install mzsae"
    pw = draw.textlength(pill_str, font=font_mono_pill)
    draw.rounded_rectangle([60, 532, 60 + pw + 32, 574], radius=21, fill="#F0EDE6", outline="#E4E0D6", width=1)
    draw.text((76, 544), "$ pip install mzsae", fill="#191919", font=font_mono_pill)

    # Colophon / Link
    colophon = "mohamedhossammohamed.github.io/MZSAE  •  Apache 2.0"
    cw = draw.textlength(colophon, font=font_sans_bold)
    draw.text((W - 60 - cw, 545), colophon, fill="#8A867D", font=font_sans_bold)

    os.makedirs("landing/assets", exist_ok=True)
    os.makedirs("docs/assets", exist_ok=True)
    img.save("landing/assets/social_preview.png", "PNG", optimize=True)
    img.save("docs/assets/social_preview.png", "PNG", optimize=True)
    print("Social preview card saved to landing/assets/social_preview.png (1200x630)")

if __name__ == "__main__":
    generate()
