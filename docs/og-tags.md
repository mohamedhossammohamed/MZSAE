# Open Graph & Twitter Card Metadata Specification

To ensure that links to MZSAE generate crisp, rich, and high-conversion preview cards across platforms (X / Twitter, LinkedIn, Slack, Discord, Facebook, iMessage), use the following metadata standard.

## HTML Meta Tags (Injected into `<head>`)

```html
<!-- Primary Meta Tags -->
<title>MZSAE — Neuromorphic Sparse Attention for Edge Inference</title>
<meta name="title" content="MZSAE — Neuromorphic Sparse Attention for Edge Inference">
<meta name="description" content="MZSAE is a biologically-inspired sparse attention engine that breaks the memory bandwidth wall on Apple Silicon. 8.4× faster decoding at 11.6k context with bit-exact outputs.">
<meta name="author" content="Mohammed Hossam Zahran">
<meta name="keywords" content="Sparse Attention, Neuromorphic Computing, Apple Silicon, Metal, LLM Inference, KV Cache Compression, CUDA, Deep Learning">
<link rel="canonical" href="https://mohamedhossammohamed.github.io/MZSAE/">

<!-- Open Graph / Facebook / LinkedIn / Discord / Slack / iMessage -->
<meta property="og:type" content="website">
<meta property="og:url" content="https://mohamedhossammohamed.github.io/MZSAE/">
<meta property="og:site_name" content="MZSAE">
<meta property="og:title" content="MZSAE — Neuromorphic Sparse Attention">
<meta property="og:description" content="Attention that reads only what matters. 8.4× faster decode at 11.6k context, 96.7% DRAM pruned, bit-exact retrieval.">
<meta property="og:image" content="https://mohamedhossammohamed.github.io/MZSAE/assets/social_preview.png">
<meta property="og:image:secure_url" content="https://mohamedhossammohamed.github.io/MZSAE/assets/social_preview.png">
<meta property="og:image:type" content="image/png">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta property="og:image:alt" content="MZSAE — Neuromorphic Sparse Attention Engine for Apple Silicon">
<meta property="og:locale" content="en_US">

<!-- Twitter / X Card -->
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:site" content="@MohamedHz72007">
<meta name="twitter:creator" content="@MohamedHz72007">
<meta name="twitter:url" content="https://mohamedhossammohamed.github.io/MZSAE/">
<meta name="twitter:title" content="MZSAE — Neuromorphic Sparse Attention">
<meta name="twitter:description" content="Attention that reads only what matters. 8.4× faster decode at 11.6k context, 96.7% DRAM pruned, bit-exact retrieval.">
<meta name="twitter:image" content="https://mohamedhossammohamed.github.io/MZSAE/assets/social_preview.png">
<meta name="twitter:image:alt" content="MZSAE — Neuromorphic Sparse Attention Engine for Apple Silicon">
```

## Schema.org JSON-LD Structured Data

```html
<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "SoftwareApplication",
  "name": "MZSAE",
  "alternateName": "Neuromorphic Sparse Attention Engine",
  "applicationCategory": "DeveloperApplication",
  "operatingSystem": "macOS, Linux",
  "author": {
    "@type": "Person",
    "name": "Mohammed Hossam Zahran",
    "url": "https://github.com/mohamedhossammohamed"
  },
  "description": "Biologically-inspired sparse attention runtime eliminating the KV-cache memory wall on Apple Silicon and modern accelerators through dual-plane memory, Cauchy-Schwarz sentinels, and TD(0) biological eviction.",
  "softwareVersion": "1.3.0",
  "license": "https://opensource.org/licenses/Apache-2.0",
  "url": "https://mohamedhossammohamed.github.io/MZSAE/",
  "downloadUrl": "https://pypi.org/project/mzsae/",
  "sameAs": [
    "https://github.com/mohamedhossammohamed/MZSAE",
    "https://x.com/MohamedHz72007"
  ]
}
</script>
```
