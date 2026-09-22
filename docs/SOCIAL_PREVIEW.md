# Social Preview Image Specification

This document details the exact specifications for the Open Graph and Twitter Card social preview image for MZSAE (`landing/assets/social_preview.png`).

## Technical Specifications

- **Dimensions**: 1200px by 630px (1.91:1 aspect ratio — standard for Twitter/X Cards and Open Graph)
- **Format**: 24-bit PNG with lossy/lossless optimization (< 100 KB)
- **Safe Zone**: Critical typography and metric cards positioned with 60px padding on all sides.
- **Generator**: Automated via `scripts/generate_social_preview.py`

## Visual Identity & Design Tokens

### Color Palette
- **Canvas**: Warm Ivory (`#FAF9F5`)
- **Card Surfaces**: Pure Paper White (`#FFFFFF`)
- **Pill & Accent Backgrounds**: Sand (`#F0EDE6`)
- **Primary Ink**: Charcoal (`#191919`)
- **Secondary Ink**: Muted Charcoal (`#55524B`)
- **Tertiary & Muted**: Stone Gray (`#8A867D`)
- **Accent Brand**: Terracotta (`#C96442`)
- **Hairlines**: Warm Gray (`#E4E0D6`)

### Typography
- **Headings & Logo**: [Source Serif 4](https://fonts.google.com/specimen/Source+Serif+4)
- **Body & Badges**: [Inter](https://fonts.google.com/specimen/Inter)
- **Telemetry & Numbers**: [JetBrains Mono](https://fonts.google.com/specimen/JetBrains+Mono)

## Layer Breakdown

1. **Top Header Bar**:
   - Logo: `MZSAE` (42px Source Serif 4, `#191919` + `#C96442`)
   - Badge: `NEUROMORPHIC SPARSE ATTENTION · V1.3.0` (Pill border `#E4E0D6`, `#C96442` font)
2. **Main Headline**:
   - Eyebrow: `EDGE INFERENCE • APPLE SILICON & ACCELERATORS` (`#8A867D`)
   - Title: `Attention that reads only what matters.` (52px Source Serif 4)
   - Lede: Autoregressive decoding memory bus resolution summary.
3. **Metric Cards Grid (4 Columns)**:
   - `8.4×`: Faster Decode (vs dense SDPA at 11.6k)
   - `96.7%`: DRAM Pruned (payload traffic in L2)
   - `10/10`: Bit-Exact Retrieval (dual-panel verified)
   - `3.28×`: KV Compression (2-bit residual cache)
4. **Bottom Colophon**:
   - Pill: `$ pip install mzsae`
   - Repository: `mohamedhossammohamed.github.io/MZSAE • Apache 2.0`

## Generation

To regenerate the card from source:
```bash
./.venv/bin/python scripts/generate_social_preview.py
```
