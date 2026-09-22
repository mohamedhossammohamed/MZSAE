# Social Preview Image Specification

This document details the exact specifications for generating the GitHub Open Graph social preview image for MZSAE.

## Technical Specifications

- **Dimensions**: 1280px by 640px (2:1 aspect ratio)
- **Format**: PNG (preferred for crisp text and sharp lines) or SVG (for vectors)
- **Safe Zone**: Keep critical elements within the center 1100x500px area to account for cropping on various platforms.

## Visual Identity

### Color Palette
- **Background**: Deep Space Blue (`#0D1117`)
- **Primary Accents**: Electric Cyan (`#58A6FF`)
- **Secondary Accents**: Neural Purple (`#A371F7`)
- **Text (Primary)**: Pure White (`#FFFFFF`)
- **Text (Secondary)**: Slate Gray (`#8B949E`)

### Typography
- **Headings & Body**: [Inter](https://fonts.google.com/specimen/Inter)
- **Metrics & Code**: [JetBrains Mono](https://fonts.google.com/specimen/JetBrains+Mono)

## Figma-Style Layer Specification (Top to Bottom)

1. **Text Overlay Layer**
   - **Logo/Title**: `MZSAE` (Inter Bold, 84px, `#FFFFFF`, Left-aligned: X=120px, Y=220px)
   - **Subtitle**: `Neuromorphic Sparse Attention Engine` (Inter SemiBold, 42px, `#58A6FF`, Left-aligned: X=120px, Y=320px)
   - **Tagline**: `Apple Silicon Native • Bio-Inspired • Extreme Context` (Inter Regular, 28px, `#8B949E`, Left-aligned: X=120px, Y=390px)
2. **Metrics Display Layer (Bottom-Aligned Grid)**
   - **Metric 1**: `6.50x` (JetBrains Mono Bold, `#A371F7`, 48px) | `Speedup vs MLX SDPA`
   - **Metric 2**: `95.9%` (JetBrains Mono Bold, `#A371F7`, 48px) | `Block Pruning`
   - **Metric 3**: `83.9%` (JetBrains Mono Bold, `#A371F7`, 48px) | `DRAM Reduction`
   - **Positioning**: Horizontal grid starting at X=120px, Y=520px
3. **Graphics Layer**
   - **Circuit Motif**: Stylized neural network nodes connected by lines, overlaying the right side of the canvas. Gradient stroke from `#58A6FF` to `#A371F7`.
4. **Background Layer**
   - Solid fill `#0D1117`.
   - Subtle radial gradient in the top right corner (`#A371F7` at 15% opacity fading to transparent).

## Complete SVG Template

You can save the following code as an `.svg` file and open it in a browser or vector editor to see the generated social preview template.

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1280 640" width="1280" height="640">
  <defs>
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&amp;family=JetBrains+Mono:wght@700&amp;display=swap');
      .title { font-family: 'Inter', sans-serif; font-weight: 700; font-size: 84px; fill: #FFFFFF; }
      .subtitle { font-family: 'Inter', sans-serif; font-weight: 600; font-size: 42px; fill: #58A6FF; }
      .tagline { font-family: 'Inter', sans-serif; font-weight: 400; font-size: 28px; fill: #8B949E; }
      .metric-value { font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 48px; fill: #A371F7; }
      .metric-label { font-family: 'Inter', sans-serif; font-weight: 400; font-size: 20px; fill: #8B949E; }
    </style>
    <radialGradient id="glow" cx="80%" cy="20%" r="60%">
      <stop offset="0%" stop-color="#A371F7" stop-opacity="0.15" />
      <stop offset="100%" stop-color="#0D1117" stop-opacity="0" />
    </radialGradient>
    <linearGradient id="line-grad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#58A6FF" />
      <stop offset="100%" stop-color="#A371F7" />
    </linearGradient>
  </defs>

  <!-- Background -->
  <rect width="1280" height="640" fill="#0D1117" />
  <rect width="1280" height="640" fill="url(#glow)" />

  <!-- Neural/Circuit Motif (Background right) -->
  <g stroke="url(#line-grad)" stroke-width="2" fill="none" opacity="0.6">
    <!-- Lines -->
    <path d="M 800,100 L 950,200 L 1100,150" />
    <path d="M 950,200 L 1050,350 L 1200,300" />
    <path d="M 950,200 L 900,400 L 1150,500" />
    <path d="M 1050,350 L 1150,500" />
    <!-- Nodes -->
    <circle cx="800" cy="100" r="6" fill="#58A6FF" stroke="none" />
    <circle cx="950" cy="200" r="8" fill="#A371F7" stroke="none" />
    <circle cx="1100" cy="150" r="6" fill="#58A6FF" stroke="none" />
    <circle cx="1050" cy="350" r="8" fill="#A371F7" stroke="none" />
    <circle cx="1200" cy="300" r="5" fill="#58A6FF" stroke="none" />
    <circle cx="900" cy="400" r="6" fill="#58A6FF" stroke="none" />
    <circle cx="1150" cy="500" r="7" fill="#A371F7" stroke="none" />
  </g>

  <!-- Text Content -->
  <text x="120" y="220" class="title">MZSAE</text>
  <text x="120" y="300" class="subtitle">Neuromorphic Sparse Attention Engine</text>
  <text x="120" y="360" class="tagline">Apple Silicon Native • Bio-Inspired • Extreme Context</text>

  <!-- Metrics Grid -->
  <g transform="translate(120, 520)">
    <!-- Metric 1 -->
    <text x="0" y="0" class="metric-value">6.50x</text>
    <text x="0" y="30" class="metric-label">Speedup vs MLX SDPA</text>
    
    <!-- Metric 2 -->
    <text x="350" y="0" class="metric-value">95.9%</text>
    <text x="350" y="30" class="metric-label">Block Pruning</text>
    
    <!-- Metric 3 -->
    <text x="650" y="0" class="metric-value">83.9%</text>
    <text x="650" y="30" class="metric-label">DRAM Reduction</text>
  </g>
</svg>
```
