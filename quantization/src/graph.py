"""Artifact-backed HG-PIPE integer graph reconstruction.

This module composes the public ICCAD24-HG-PIPE reference artifacts into
larger graph-level checks.  It uses the saved integer weights, bias terms,
scale factors, and lookup tables instead of PyTorch fake quantization.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .artifacts import HgPipeSource, read_ints
from .ops import layernorm_quantize, softmax_quantize, table_quantize


DEFAULT_TOKENS = 196
DEFAULT_CHANNELS = 192
DEFAULT_HEADS = 3


@dataclass(frozen=True)
class GraphVerificationResult:
    name: str
    kind: str
    passed: bool
    elements: int
    mismatches: int
    max_abs_error: int


def _as_array(values: Iterable[int]) -> np.ndarray:
    return np.asarray(list(values), dtype=np.int64)


def _compare(name: str, kind: str, actual: np.ndarray, expected: np.ndarray) -> GraphVerificationResult:
    actual = np.asarray(actual, dtype=np.int64).reshape(-1)
    expected = np.asarray(expected, dtype=np.int64).reshape(-1)
    shared = min(actual.size, expected.size)
    mismatches = abs(actual.size - expected.size)
    max_abs_error = 0
    if shared:
        diff = actual[:shared] - expected[:shared]
        mismatches += int(np.count_nonzero(diff))
        max_abs_error = int(np.max(np.abs(diff)))
    return GraphVerificationResult(
        name=name,
        kind=kind,
        passed=mismatches == 0,
        elements=int(expected.size),
        mismatches=mismatches,
        max_abs_error=max_abs_error,
    )


class ArtifactGraphRunner:
    """Run graph-level integer reconstruction against `case/refs` artifacts."""

    def __init__(self, source: HgPipeSource | str | Path):
        if not isinstance(source, HgPipeSource):
            source = HgPipeSource.from_path(source)
        self.source = source
        self.refs = source.refs

    def ints(self, name: str) -> list[int]:
        return read_ints(self.refs / name)

    def array(self, name: str) -> np.ndarray:
        return _as_array(self.ints(name))

    def static_matmul(self, stem: str, input_values: np.ndarray | None = None) -> np.ndarray:
        x = self.array(f"{stem}_input.txt") if input_values is None else np.asarray(input_values, dtype=np.int64)
        weight = self.array(f"{stem}_weight.txt")
        bias = self.array(f"{stem}_bias.txt")

        co = bias.size
        ci = weight.size // co
        if x.size % ci:
            raise ValueError(f"{stem}: input length {x.size} is not divisible by inferred CI={ci}")
        tokens = x.size // ci
        y = x.reshape(tokens, ci) @ weight.reshape(co, ci).T
        y += bias.reshape(1, co)
        return y.reshape(-1)

    def dynamic_head_matmul(
        self,
        stem: str,
        *,
        heads: int = DEFAULT_HEADS,
        tokens: int = DEFAULT_TOKENS,
        ci: int,
        co: int,
        input_values: np.ndarray | None = None,
        weight_values: np.ndarray | None = None,
    ) -> np.ndarray:
        x = self.array(f"{stem}_input.txt") if input_values is None else np.asarray(input_values, dtype=np.int64)
        weight = self.array(f"{stem}_weight.txt") if weight_values is None else np.asarray(weight_values, dtype=np.int64)
        x = x.reshape(heads, tokens, ci)
        weight = weight.reshape(heads, co, ci)
        return np.concatenate([(x[head] @ weight[head].T).reshape(-1) for head in range(heads)])

    @staticmethod
    def split_heads(values: np.ndarray, *, tokens: int = DEFAULT_TOKENS, channels: int = DEFAULT_CHANNELS, heads: int = DEFAULT_HEADS) -> np.ndarray:
        x = np.asarray(values, dtype=np.int64).reshape(tokens, channels)
        channels_per_head = channels // heads
        return np.concatenate([x[:, head * channels_per_head : (head + 1) * channels_per_head].reshape(-1) for head in range(heads)])

    @staticmethod
    def transpose_head_values(values: np.ndarray, *, tokens: int = DEFAULT_TOKENS, channels: int = DEFAULT_CHANNELS, heads: int = DEFAULT_HEADS) -> np.ndarray:
        x = np.asarray(values, dtype=np.int64).reshape(tokens, channels)
        channels_per_head = channels // heads
        return np.concatenate([x[:, head * channels_per_head : (head + 1) * channels_per_head].T.reshape(-1) for head in range(heads)])

    @staticmethod
    def merge_heads(values: np.ndarray, *, tokens: int = DEFAULT_TOKENS, channels: int = DEFAULT_CHANNELS, heads: int = DEFAULT_HEADS) -> np.ndarray:
        channels_per_head = channels // heads
        x = np.asarray(values, dtype=np.int64).reshape(heads, tokens, channels_per_head)
        return x.transpose(1, 0, 2).reshape(tokens, channels).reshape(-1)

    def table_quant(self, scalar_name: str, table_name: str, input_values: np.ndarray) -> np.ndarray:
        return _as_array(table_quantize(input_values.reshape(-1).tolist(), self.ints(scalar_name), self.ints(table_name)))

    def layernorm(self, stem: str, input_values: np.ndarray | None = None) -> np.ndarray:
        x = self.array(f"{stem}_input.txt") if input_values is None else np.asarray(input_values, dtype=np.int64)
        return _as_array(
            layernorm_quantize(
                x.reshape(-1).tolist(),
                self.ints(f"{stem}_scalars.txt"),
                self.ints(f"{stem}_lnw_m.txt"),
                self.ints(f"{stem}_lnb_m.txt"),
                self.ints(f"{stem}_rsqrt_table_m.txt"),
            )
        )

    def softmax(self, stem: str, input_values: np.ndarray | None = None, *, tokens: int = DEFAULT_TOKENS, heads: int = DEFAULT_HEADS) -> np.ndarray:
        x = self.array(f"{stem}_input.txt") if input_values is None else np.asarray(input_values, dtype=np.int64)
        return _as_array(
            softmax_quantize(
                x.reshape(-1).tolist(),
                self.ints(f"{stem}_scalars.txt"),
                self.ints(f"{stem}_exp_opp_table_m.txt"),
                self.ints(f"{stem}_recip_scaled_table_m_one.txt"),
                self.ints(f"{stem}_recip_scaled_table_m_two.txt"),
                tokens=tokens,
                heads=heads,
            )
        )

    def residual_merge(self, block_stem: str, main_values: np.ndarray, residual_values: np.ndarray | None = None) -> np.ndarray:
        residual = self.array(f"{block_stem}_input.txt") if residual_values is None else np.asarray(residual_values, dtype=np.int64)
        rm, rs = self.ints(f"{block_stem}_scalars.txt")[:2]
        rb = 1 << (rs - 1)
        return np.asarray(main_values, dtype=np.int64) + ((residual * rm + rb) >> rs)

    def patch_embed(self, input_values: np.ndarray | None = None) -> np.ndarray:
        stem = "patch_embed_matmul"
        x = self.array(f"{stem}_input.txt") if input_values is None else np.asarray(input_values, dtype=np.int64)
        weight = self.array(f"{stem}_weight.txt")
        bias = self.array(f"{stem}_bias.txt")
        cls = self.array("patch_embed_cls.txt")
        shift = self.ints("patch_embed_scalars.txt")[0]
        co = cls.size
        ci = weight.size // co
        tokens = x.size // ci
        y = x.reshape(tokens, ci) @ weight.reshape(co, ci).T
        y += bias.reshape(tokens, co)
        y[0, :] = cls
        return ((y + (1 << (shift - 1))) >> shift).reshape(-1)

    def mlp_block(self, index: int, input_values: np.ndarray | None = None) -> tuple[np.ndarray, list[GraphVerificationResult]]:
        prefix = f"mlp_{index}"
        results: list[GraphVerificationResult] = []
        x = self.array(f"{prefix}_input.txt") if input_values is None else np.asarray(input_values, dtype=np.int64)
        ln = self.layernorm(f"{prefix}_lnq", x)
        results.append(_compare(f"{prefix}_lnq", "graph_layernorm", ln, self.array(f"{prefix}_lnq_output.txt")))
        m1 = self.static_matmul(f"{prefix}_matmul1", ln)
        results.append(_compare(f"{prefix}_matmul1", "graph_matmul", m1, self.array(f"{prefix}_matmul1_output.txt")))
        ge = self.table_quant(f"{prefix}_geluq_scalars.txt", f"{prefix}_geluq_table_m.txt", m1)
        results.append(_compare(f"{prefix}_geluq", "graph_gelu_table", ge, self.array(f"{prefix}_geluq_output.txt")))
        m2 = self.static_matmul(f"{prefix}_matmul2", ge)
        results.append(_compare(f"{prefix}_matmul2", "graph_matmul", m2, self.array(f"{prefix}_matmul2_output.txt")))
        out = self.residual_merge(prefix, m2, x)
        results.append(_compare(prefix, "graph_mlp_block", out, self.array(f"{prefix}_output.txt")))
        return out, results

    def attention_block(self, index: int, input_values: np.ndarray | None = None) -> tuple[np.ndarray, list[GraphVerificationResult]]:
        prefix = f"attn_{index}"
        results: list[GraphVerificationResult] = []
        x = self.array(f"{prefix}_input.txt") if input_values is None else np.asarray(input_values, dtype=np.int64)
        ln = self.layernorm(f"{prefix}_lnq", x)
        results.append(_compare(f"{prefix}_lnq", "graph_layernorm", ln, self.array(f"{prefix}_lnq_output.txt")))

        q = self.static_matmul(f"{prefix}_gen_q_matmul", ln)
        k = self.static_matmul(f"{prefix}_gen_k_matmul", ln)
        v = self.static_matmul(f"{prefix}_gen_v_matmul", ln)
        results.append(_compare(f"{prefix}_gen_q_matmul", "graph_matmul", q, self.array(f"{prefix}_gen_q_matmul_output.txt")))
        results.append(_compare(f"{prefix}_gen_k_matmul", "graph_matmul", k, self.array(f"{prefix}_gen_k_matmul_output.txt")))
        results.append(_compare(f"{prefix}_gen_v_matmul", "graph_matmul", v, self.array(f"{prefix}_gen_v_matmul_output.txt")))

        qq = self.table_quant(f"{prefix}_q_q_scalars.txt", f"{prefix}_q_q_table_m.txt", q)
        kq = self.table_quant(f"{prefix}_k_q_scalars.txt", f"{prefix}_k_q_table_m.txt", k)
        vq = self.table_quant(f"{prefix}_v_q_scalars.txt", f"{prefix}_v_q_table_m.txt", v)
        results.append(_compare(f"{prefix}_qq", "graph_requant_table", qq, self.array(f"{prefix}_qq_output.txt")))
        results.append(_compare(f"{prefix}_kq", "graph_requant_table", kq, self.array(f"{prefix}_kq_output.txt")))
        results.append(_compare(f"{prefix}_vq", "graph_requant_table", vq, self.array(f"{prefix}_vq_output.txt")))

        q_heads = self.split_heads(qq)
        k_heads = self.split_heads(kq)
        v_heads_t = self.transpose_head_values(vq)
        results.append(_compare(f"{prefix}_gen_r_matmul_input", "graph_head_split", q_heads, self.array(f"{prefix}_gen_r_matmul_input.txt")))
        results.append(_compare(f"{prefix}_gen_r_matmul_weight", "graph_head_split", k_heads, self.array(f"{prefix}_gen_r_matmul_weight.txt")))
        results.append(_compare(f"{prefix}_gen_a_matmul_weight", "graph_head_transpose", v_heads_t, self.array(f"{prefix}_gen_a_matmul_weight.txt")))

        r = self.dynamic_head_matmul(f"{prefix}_gen_r_matmul", ci=64, co=DEFAULT_TOKENS, input_values=q_heads, weight_values=k_heads)
        results.append(_compare(f"{prefix}_gen_r_matmul", "graph_dynamic_matmul", r, self.array(f"{prefix}_gen_r_matmul_output.txt")))
        softmax = self.softmax(f"{prefix}_softmaxq", r)
        results.append(_compare(f"{prefix}_softmaxq", "graph_softmax_table", softmax, self.array(f"{prefix}_softmaxq_output.txt")))
        a = self.dynamic_head_matmul(f"{prefix}_gen_a_matmul", ci=DEFAULT_TOKENS, co=64, input_values=softmax, weight_values=v_heads_t)
        results.append(_compare(f"{prefix}_gen_a_matmul", "graph_dynamic_matmul", a, self.array(f"{prefix}_gen_a_matmul_output.txt")))
        aq = self.table_quant(f"{prefix}_a_q_scalars.txt", f"{prefix}_a_q_table_m.txt", a)
        results.append(_compare(f"{prefix}_aq", "graph_requant_table", aq, self.array(f"{prefix}_aq_output.txt")))
        o_input = self.merge_heads(aq)
        results.append(_compare(f"{prefix}_gen_o_matmul_input", "graph_head_merge", o_input, self.array(f"{prefix}_gen_o_matmul_input.txt")))
        o = self.static_matmul(f"{prefix}_gen_o_matmul", o_input)
        results.append(_compare(f"{prefix}_gen_o_matmul", "graph_matmul", o, self.array(f"{prefix}_gen_o_matmul_output.txt")))
        out = self.residual_merge(prefix, o, x)
        results.append(_compare(prefix, "graph_attention_block", out, self.array(f"{prefix}_output.txt")))
        return out, results

    def head(self, input_values: np.ndarray | None = None) -> tuple[np.ndarray, list[GraphVerificationResult]]:
        results: list[GraphVerificationResult] = []
        x = self.array("head_input.txt") if input_values is None else np.asarray(input_values, dtype=np.int64)
        cls = x.reshape(DEFAULT_TOKENS, DEFAULT_CHANNELS)[0]
        ln = self.layernorm("head_lnq", cls)
        results.append(_compare("head_lnq", "graph_layernorm", ln, self.array("head_lnq_output.txt")))
        logits = self.static_matmul("head_matmul", ln)
        results.append(_compare("head_matmul", "graph_matmul", logits, self.array("head_matmul_output.txt")))
        results.append(_compare("head", "graph_head", logits, self.array("head_output.txt")))
        return logits, results

    def forward_from_patch_input(self, input_values: np.ndarray | None = None) -> tuple[np.ndarray, list[GraphVerificationResult]]:
        results: list[GraphVerificationResult] = []
        current = self.patch_embed(input_values)
        results.append(_compare("patch_embed", "graph_patch_embed", current, self.array("patch_embed_output.txt")))
        results.append(_compare("patch_embed_to_attn_0", "graph_chain_boundary", current, self.array("attn_0_input.txt")))

        for index in range(12):
            current, attn_results = self.attention_block(index, current)
            results.extend(attn_results)
            results.append(_compare(f"attn_{index}_to_mlp_{index}", "graph_chain_boundary", current, self.array(f"mlp_{index}_input.txt")))

            current, mlp_results = self.mlp_block(index, current)
            results.extend(mlp_results)
            if index < 11:
                results.append(_compare(f"mlp_{index}_to_attn_{index + 1}", "graph_chain_boundary", current, self.array(f"attn_{index + 1}_input.txt")))
            else:
                results.append(_compare("mlp_11_to_head", "graph_chain_boundary", current, self.array("head_input.txt")))

        logits, head_results = self.head(current)
        results.extend(head_results)
        return logits, results

    def verify_end_to_end(self, input_values: np.ndarray | None = None) -> list[GraphVerificationResult]:
        _, results = self.forward_from_patch_input(input_values)
        return results

    def verify_graph(self, *, blocks: Iterable[int] | None = None) -> list[GraphVerificationResult]:
        if blocks is None:
            blocks = range(12)
        results: list[GraphVerificationResult] = []
        results.append(_compare("patch_embed", "graph_patch_embed", self.patch_embed(), self.array("patch_embed_output.txt")))
        for index in blocks:
            if (self.refs / f"attn_{index}_input.txt").exists():
                _, attn_results = self.attention_block(index)
                results.extend(attn_results)
            if (self.refs / f"mlp_{index}_input.txt").exists():
                _, mlp_results = self.mlp_block(index)
                results.extend(mlp_results)
        _, head_results = self.head()
        results.extend(head_results)
        return results
