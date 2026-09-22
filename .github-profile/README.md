<div align="center">

```
 ███╗   ███╗ ███████╗ ███████╗  █████╗  ███████╗
 ████╗ ████║ ╚══███╔╝ ██╔════╝ ██╔══██╗ ██╔════╝
 ██╔████╔██║   ███╔╝  ███████╗ ███████║ █████╗
 ██║╚██╔╝██║  ███╔╝   ╚════██║ ██╔══██║ ██╔══╝
 ██║ ╚═╝ ██║ ███████╗ ███████║ ██║  ██║ ███████╗
 ╚═╝     ╚═╝ ╚══════╝ ╚══════╝ ╚═╝  ╚═╝ ╚══════╝
```

### Mohammed Hossam Zahran

**Systems engineer building at the intersection of neuroscience, GPU microarchitecture, and LLM inference.**

*Neuromorphic Sparse Attention Engine — breaking the Apple Silicon memory wall*

---

[![GitHub](https://img.shields.io/badge/GitHub-mohamedhossammohamed-181717?style=flat&logo=github)](https://github.com/mohamedhossammohamed)
[![X / Twitter](https://img.shields.io/badge/@MohamedHz72007-000000?style=flat&logo=x&logoColor=white)](https://twitter.com/MohamedHz72007)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat)](https://opensource.org/licenses/MIT)

</div>

---

## 🔬 Currently Building

<table>
<tr>
<td width="80" align="center">
<img src="https://img.shields.io/badge/MZSAE-v1.2.0-58A6FF?style=for-the-badge&labelColor=0D1117" alt="MZSAE v1.2.0"/>
</td>
<td>

**[MZSAE — Neuromorphic Sparse Attention Engine](https://github.com/mohamedhossammohamed/MZSAE)**

Hardware-sympathetic sparse attention that decouples RoPE into fast/slow manifolds, constructs L2-resident Cauchy-Schwarz sentinels, and uses neuromorphic TD(0) eviction to achieve infinite effective context on Apple Silicon edge devices.

| Metric | Value |
|:---|:---|
| **Speed vs MLX SDPA (128k)** | **6.50×** faster |
| **Block Pruning Ratio** | **95.9%** of DRAM blocks skipped |
| **KV Cache Compression** | **3.28×** (128 MB → 39 MB at 128k) |
| **NIAH Accuracy (16k, budgeted)** | **100%** (vs 0% dense+FIFO collapse) |
| **Cosine Fidelity vs Dense FP16** | **0.9968** |
| **E2E Decode (Qwen2.5-0.5B, M4)** | **~63 tok/s** |

</td>
</tr>
</table>

---

## 🛠 Tech Stack

<p align="center">
<img src="https://img.shields.io/badge/Metal-MSL_3.1-A371F7?style=for-the-badge&logo=apple&logoColor=white" alt="Metal"/>
<img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/>
<img src="https://img.shields.io/badge/Apple_Silicon-M1/M4-333333?style=for-the-badge&logo=apple&logoColor=white" alt="Apple Silicon"/>
<img src="https://img.shields.io/badge/NumPy-1.24+-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy"/>
<img src="https://img.shields.io/badge/MLX-Integration-58A6FF?style=for-the-badge&logo=apple&logoColor=white" alt="MLX"/>
<img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
<img src="https://img.shields.io/badge/C++-Metal_Runtime-00599C?style=for-the-badge&logo=cplusplus&logoColor=white" alt="C++"/>
</p>

---

## 📊 GitHub Stats

<div align="center">
<a href="https://github.com/mohamedhossammohamed">
<img height="180em" src="https://github-readme-stats.vercel.app/api?username=mohamedhossammohamed&show_icons=true&theme=github_dark&include_all_commits=true&count_private=true&hide_border=true&bg_color=0D1117&title_color=58A6FF&icon_color=A371F7&text_color=C9D1D9" alt="GitHub Stats"/>
<img height="180em" src="https://github-readme-stats.vercel.app/api/top-langs/?username=mohamedhossammohamed&layout=compact&theme=github_dark&hide_border=true&bg_color=0D1117&title_color=58A6FF&text_color=C9D1D9&langs_count=8" alt="Top Languages"/>
</a>
</div>

<div align="center">
<img src="https://github-readme-activity-graph.vercel.app/graph?username=mohamedhossammohamed&bg_color=0D1117&color=C9D1D9&line=58A6FF&point=A371F7&area=true&area_color=58A6FF&hide_border=true" alt="Activity Graph" width="100%"/>
</div>

---

## 📌 Featured Repository

<div align="center">
<a href="https://github.com/mohamedhossammohamed/MZSAE">
<img src="https://github-readme-stats.vercel.app/api/pin/?username=mohamedhossammohamed&repo=MZSAE&theme=github_dark&hide_border=true&bg_color=0D1117&title_color=58A6FF&icon_color=A371F7&text_color=C9D1D9" alt="MZSAE"/>
</a>
</div>

---

## 🧠 Research Interests

- **Sparse Attention Mechanisms** — Cauchy-Schwarz bounds, sentinel pruning, memory-bandwidth-optimal tiling
- **Neuromorphic Computing** — TD(0) reinforcement learning for cache eviction, bio-inspired memory consolidation
- **GPU Microarchitecture** — Apple Metal MSL 3.1, CUDA SM90, register-level optimization, GQA threadgroup fusion
- **LLM Inference** — KV cache compression, 4-bit quantization, infinite-context edge deployment

---

<div align="center">

*"We'd rather fix the narrative now than defend hype later."*

— [MZSAE LIMITATIONS.md](https://github.com/mohamedhossammohamed/MZSAE/blob/main/docs/LIMITATIONS.md)

</div>
