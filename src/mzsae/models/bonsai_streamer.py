"""
BonsaiWeightStreamer: Model-scoped weight streaming coordinator for Bonsai models.
Supports BonsAI 1.7B (Qwen3 architecture, PQ2_0 ternary) and BonsAI 27B / DSpark (Q4_1 / PRISM_Q2_0).
MZahran Sparse Attention Engine (MZSAE)
"""

from __future__ import annotations

import mmap
import os
import struct
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np


def parse_bonsai_layer_index(name: str) -> int:
    """
    Parses layer index from tensor name supporting:
    - 'model.layers.N.'
    - 'blk.N.'
    Returns -1 for shared tensors (e.g. token_embd, output_norm, etc.)
    """
    prefixes = ["model.layers.", "blk."]
    matched_prefix = None
    for p in prefixes:
        if name.startswith(p):
            matched_prefix = p
            break
    if not matched_prefix:
        return -1

    rest = name[len(matched_prefix):]
    digits = ""
    for ch in rest:
        if ch.isdigit():
            digits += ch
        else:
            break
    if not digits or not rest[len(digits):].startswith("."):
        return -1
    return int(digits)


@dataclass
class BonsaiTensorEntry:
    name: str
    offset: int
    nbytes: int
    dims: List[int]
    ttype: int


class BonsaiWeightStreamer:
    """Zero-copy mmap layer streamer for Bonsai GGUF weights."""

    def __init__(self, path: str | Path, resident_layer_count: int = 2) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Model file not found: {self.path}")
        self.resident_layer_count = int(resident_layer_count)
        self._f = open(self.path, "rb")
        self._mm = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)
        self.tensors: Dict[str, BonsaiTensorEntry] = {}
        self.metadata: Dict[str, Any] = {}
        self.groups: Dict[int, List[str]] = {}
        self._resident: OrderedDict[int, None] = OrderedDict()
        self._parse()

        self._shared_bytes = sum(self.tensors[n].nbytes for n in self.groups.get(-1, []))
        self._current_bytes = self._shared_bytes
        self._peak_bytes = self._current_bytes
        self._advise_shared()

    def _parse(self) -> None:
        buf = self._mm
        if bytes(buf[0:4]) != b"GGUF":
            raise ValueError("Bad GGUF magic")
        version = struct.unpack_from("<I", buf, 4)[0]
        if version not in (2, 3):
            raise ValueError(f"Unsupported GGUF version {version}")
        n_tensors = struct.unpack_from("<Q", buf, 8)[0]
        n_kv = struct.unpack_from("<Q", buf, 16)[0]

        alignment = 32
        off = 24

        def _str(o: int) -> Tuple[str, int]:
            n = struct.unpack_from("<Q", buf, o)[0]
            o += 8
            return buf[o:o + n].decode("utf-8", "replace"), o + n

        def _read_val(o: int, vt: int) -> Tuple[Any, int]:
            if vt == 0: return struct.unpack_from("<B", buf, o)[0], o + 1
            if vt == 1: return struct.unpack_from("<b", buf, o)[0], o + 1
            if vt == 2: return struct.unpack_from("<H", buf, o)[0], o + 2
            if vt == 3: return struct.unpack_from("<h", buf, o)[0], o + 2
            if vt == 4: return struct.unpack_from("<I", buf, o)[0], o + 4
            if vt == 5: return struct.unpack_from("<i", buf, o)[0], o + 4
            if vt == 6: return struct.unpack_from("<f", buf, o)[0], o + 4
            if vt == 7: return (struct.unpack_from("<B", buf, o)[0] != 0), o + 1
            if vt == 8:
                return _str(o)
            if vt == 9:
                it = struct.unpack_from("<I", buf, o)[0]; o += 4
                cnt = struct.unpack_from("<Q", buf, o)[0]; o += 8
                items = []
                for _ in range(cnt):
                    val, o = _read_val(o, it)
                    items.append(val)
                return items, o
            if vt in (10, 11, 12): return struct.unpack_from("<Q", buf, o)[0], o + 8
            raise ValueError(f"Unknown vtype {vt}")

        for _ in range(n_kv):
            k, off = _str(off)
            vt = struct.unpack_from("<I", buf, off)[0]; off += 4
            val, off = _read_val(off, vt)
            self.metadata[k] = val
            if k == "general.alignment":
                alignment = val or 32

        raw_infos = []
        for _ in range(n_tensors):
            name, off = _str(off)
            n_dims = struct.unpack_from("<I", buf, off)[0]; off += 4
            dims = [struct.unpack_from("<Q", buf, off + 8 * d)[0] for d in range(n_dims)]
            off += 8 * n_dims
            ttype = struct.unpack_from("<I", buf, off)[0]; off += 4
            rel = struct.unpack_from("<Q", buf, off)[0]; off += 8
            raw_infos.append((name, dims, ttype, rel))

        data_off = ((off + alignment - 1) // alignment) * alignment
        for name, dims, ttype, rel in raw_infos:
            numel = 1
            for d in dims: numel *= d
            if ttype == 42 or ttype == 141 or ttype == 142:  # PQ2_0 / PRISM_Q2_0 (128 elements / 34 bytes)
                nbytes = (numel // 128) * 34
            elif ttype == 0:   # FP32 (4 bytes)
                nbytes = numel * 4
            elif ttype == 1:   # FP16 (2 bytes)
                nbytes = numel * 2
            elif ttype == 3:   # Q4_1
                nbytes = (numel // 32) * 20
            else:
                nbytes = (numel // 128) * 34 if (numel % 128 == 0) else numel
            self.tensors[name] = BonsaiTensorEntry(name, data_off + rel, nbytes, dims, ttype)
            layer_idx = parse_bonsai_layer_index(name)
            self.groups.setdefault(layer_idx, []).append(name)

        for names in self.groups.values():
            names.sort()

    def _advise(self, addr_obj: Any, advice: int) -> None:
        try:
            if hasattr(os, "posix_madvise"):
                os.posix_madvise(addr_obj, 0, len(addr_obj), advice)
        except Exception:
            pass

    def _advise_shared(self) -> None:
        will = getattr(os, "POSIX_MADV_WILLNEED", 3)
        for name in self.groups.get(-1, []):
            e = self.tensors[name]
            if e.nbytes:
                self._advise(memoryview(self._mm)[e.offset:e.offset + e.nbytes], will)

    @property
    def layer_indices(self) -> List[int]:
        return sorted(i for i in self.groups if i >= 0)

    @property
    def num_layers(self) -> int:
        return len(self.layer_indices)

    @property
    def total_model_bytes(self) -> int:
        return sum(e.nbytes for e in self.tensors.values())

    def get_tensor_data(self, name: str) -> memoryview:
        """Returns zero-copy memoryview into tensor weights."""
        if name not in self.tensors:
            raise KeyError(f"Tensor {name} not found in weights")
        e = self.tensors[name]
        return memoryview(self._mm)[e.offset:e.offset + e.nbytes]

    def close(self) -> None:
        try:
            self._mm.close()
        except Exception:
            pass
        finally:
            try:
                self._f.close()
            except Exception:
                pass

    def __enter__(self) -> BonsaiWeightStreamer:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
