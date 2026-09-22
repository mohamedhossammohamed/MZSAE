# Roadmap

The MZSAE project is actively evolving. Below is the planned trajectory for upcoming releases.

## Milestones

- [ ] **v1.3.0: CUDA Backend**
    - Port Metal kernels to SM90 (Hopper) architecture.
    - Leverage Tensor Cores for asynchronous TMA fetch.
- [ ] **v1.4.0: llama.cpp Integration**
    - Native attention backend patch for `llama.cpp`.
    - Provide drop-in replacement for standard edge deployments.
- [ ] **v1.5.0: Dynamic-τ ACC Veto**
    - Auto-calibrate the cosine gate threshold ($\tau$) dynamically per layer based on distribution shifts.
- [ ] **v2.0.0: Real Semantic NIAH**
    - Introduce evaluation using frozen LLM embeddings.
    - Accompanying arXiv preprint detailing neuromorphic eviction dynamics on natural language.
- [ ] **v2.1.0: Multi-turn Retention Validation**
    - Benchmarking state retention across multi-turn conversational agents with extreme memory constraints.

## Community Wishlist

We welcome contributions and feedback! Current highly requested features:
* Windows/DirectML port
* Pre-tuned hardware profiles for Snapdragon X Elite
* Hugging Face `transformers` integration layer

*Want to contribute? Check out the [GitHub repository](https://github.com/mzahran/mzsae).*
