# MZSAE

A drop-in MLX plugin for Apple Silicon that accelerates long-context autoregressive decoding by pruning the KV cache via a learned block-sparse policy, plus a lossless ternary weight compression component.

This repository distributes **compiled builds only** (Python wheels and compiled Metal libraries). Source is not published here.

## Status

Under active development. No release has been published yet — check the [Releases](../../releases) page for builds once available, and `RESULT.md` in a future release for independently measured decode benchmarks (model, exact baseline, and hardware labeled per figure — no numbers are published until they're verified).

## Install

Once a release is published:

```bash
pip install mzsae
```

## License

Distribution of compiled builds from this repository is governed by the license included with each release. This is not an open-source release of the MZSAE source code.
