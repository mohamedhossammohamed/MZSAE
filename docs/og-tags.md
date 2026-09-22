# Open Graph Metadata & SEO

To ensure that links to the MZSAE repository or documentation site generate rich, accurate previews on platforms like Twitter, Slack, and Discord, use the following Open Graph, Twitter Card, and Schema.org metadata.

## Implementation Instructions

Insert the following HTML `<meta>` tags and JSON-LD scripts inside the `<head>` section of your documentation site (e.g., GitHub Pages, Sphinx docs, or Docusaurus). 

### HTML Meta Tags (Open Graph & Twitter Card)

```html
<!-- Primary Meta Tags -->
<title>MZSAE - Neuromorphic Sparse Attention Engine</title>
<meta name="title" content="MZSAE - Neuromorphic Sparse Attention Engine">
<meta name="description" content="A bio-inspired sparse attention engine optimized for Apple Silicon. Achieves 6.50x speedup and 83.9% DRAM reduction at 128k context with 100% NIAH accuracy.">

<!-- Open Graph / Facebook / LinkedIn / Slack -->
<meta property="og:type" content="website">
<meta property="og:url" content="https://github.com/mohamedhossammohamed/MZSAE">
<meta property="og:title" content="MZSAE - Neuromorphic Sparse Attention Engine">
<meta property="og:description" content="A bio-inspired sparse attention engine optimized for Apple Silicon. Achieves 6.50x speedup and 83.9% DRAM reduction at 128k context with 100% NIAH accuracy.">
<meta property="og:image" content="https://raw.githubusercontent.com/mohamedhossammohamed/MZSAE/main/docs/assets/social_preview.png">

<!-- Twitter -->
<meta property="twitter:card" content="summary_large_image">
<meta property="twitter:url" content="https://github.com/mohamedhossammohamed/MZSAE">
<meta property="twitter:title" content="MZSAE - Neuromorphic Sparse Attention Engine">
<meta property="twitter:description" content="A bio-inspired sparse attention engine optimized for Apple Silicon. Achieves 6.50x speedup and 83.9% DRAM reduction at 128k context with 100% NIAH accuracy.">
<meta property="twitter:image" content="https://raw.githubusercontent.com/mohamedhossammohamed/MZSAE/main/docs/assets/social_preview.png">
<meta property="twitter:creator" content="@MohamedHz72007">
<meta property="twitter:site" content="@MohamedHz72007">
```

### Schema.org JSON-LD

Schema.org JSON-LD helps search engines like Google understand the exact nature of the project. Include this right before the closing `</head>` tag.

```html
<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "SoftwareSourceCode",
  "name": "MZSAE",
  "alternateName": "Neuromorphic Sparse Attention Engine",
  "author": {
    "@type": "Person",
    "name": "Mohammed Hossam Zahran",
    "url": "https://github.com/mohamedhossammohamed"
  },
  "description": "A neuromorphic sparse attention engine for Apple Silicon Metal featuring TD(0) RL eviction, Cauchy-Schwarz sentinel pruning, and RoPE manifold decoupling. Delivers 6.50x speedup over MLX SDPA at 128k context length.",
  "programmingLanguage": ["Python", "Metal Shading Language"],
  "license": "https://opensource.org/licenses/Apache-2.0",
  "version": "1.2.0",
  "codeRepository": "https://github.com/mohamedhossammohamed/MZSAE",
  "runtimePlatform": "Apple Silicon (macOS)",
  "keywords": "Sparse Attention, Neuromorphic, Apple Silicon, MLX, PyTorch, LLM"
}
</script>
```
