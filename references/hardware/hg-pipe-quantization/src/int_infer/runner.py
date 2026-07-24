"""Artifact-backed HG-PIPE graph runner using torch integer tensors."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..artifacts import HgPipeSource, read_ints
from ..graph import DEFAULT_CHANNELS, DEFAULT_HEADS, DEFAULT_TOKENS, GraphVerificationResult, _compare
from .kernels import (
    as_int_tensor,
    dynamic_head_matmul_tensor,
    layernorm_quantize_tensor,
    merge_heads_tensor,
    residual_merge_tensor,
    softmax_quantize_tensor,
    split_heads_tensor,
    static_matmul_tensor,
    table_quantize_tensor,
    transpose_head_values_tensor,
)


class TorchIntGraphRunner:
    """Run the HG-PIPE artifact graph with torch integer operations."""

    def __init__(self, source: HgPipeSource | str | Path, *, device: str | None = None):
        if not isinstance(source, HgPipeSource):
            source = HgPipeSource.from_path(source)
        self.source = source
        self.refs = source.refs
        self.device = device

    def ints(self, name: str) -> list[int]:
        return read_ints(self.refs / name)

    def tensor(self, name: str):
        return as_int_tensor(self.ints(name), device=self.device)

    def expected(self, name: str) -> np.ndarray:
        return np.asarray(self.ints(name), dtype=np.int64)

    @staticmethod
    def numpy(values) -> np.ndarray:
        return values.detach().cpu().numpy().astype(np.int64, copy=False).reshape(-1)

    def compare(self, name: str, kind: str, actual, expected_file: str) -> GraphVerificationResult:
        return _compare(name, kind, self.numpy(actual), self.expected(expected_file))

    def static_matmul(self, stem: str, input_values=None):
        x = self.tensor(f"{stem}_input.txt") if input_values is None else input_values
        bias = self.tensor(f"{stem}_bias.txt")
        weight = self.tensor(f"{stem}_weight.txt").reshape(bias.numel(), -1)
        return static_matmul_tensor(x, weight, bias)

    def dynamic_head_matmul(self, stem: str, *, ci: int, co: int, input_values=None, weight_values=None):
        x = self.tensor(f"{stem}_input.txt") if input_values is None else input_values
        weight = self.tensor(f"{stem}_weight.txt") if weight_values is None else weight_values
        return dynamic_head_matmul_tensor(x, weight, heads=DEFAULT_HEADS, tokens=DEFAULT_TOKENS, ci=ci, co=co)

    def table_quant(self, scalar_name: str, table_name: str, input_values):
        return table_quantize_tensor(input_values, self.ints(scalar_name), self.ints(table_name))

    def layernorm(self, stem: str, input_values=None):
        x = self.tensor(f"{stem}_input.txt") if input_values is None else input_values
        return layernorm_quantize_tensor(
            x,
            self.ints(f"{stem}_scalars.txt"),
            self.tensor(f"{stem}_lnw_m.txt"),
            self.tensor(f"{stem}_lnb_m.txt"),
            self.tensor(f"{stem}_rsqrt_table_m.txt"),
        )

    def softmax(self, stem: str, input_values=None):
        x = self.tensor(f"{stem}_input.txt") if input_values is None else input_values
        return softmax_quantize_tensor(
            x,
            self.ints(f"{stem}_scalars.txt"),
            self.tensor(f"{stem}_exp_opp_table_m.txt"),
            self.tensor(f"{stem}_recip_scaled_table_m_one.txt"),
            self.tensor(f"{stem}_recip_scaled_table_m_two.txt"),
            tokens=DEFAULT_TOKENS,
            heads=DEFAULT_HEADS,
        )

    def residual_merge(self, block_stem: str, main_values, residual_values):
        return residual_merge_tensor(main_values, residual_values, self.ints(f"{block_stem}_scalars.txt"))

    def patch_embed(self, input_values=None):
        stem = "patch_embed_matmul"
        x = self.tensor(f"{stem}_input.txt") if input_values is None else input_values
        cls = self.tensor("patch_embed_cls.txt")
        bias = self.tensor(f"{stem}_bias.txt").reshape(DEFAULT_TOKENS, DEFAULT_CHANNELS)
        weight = self.tensor(f"{stem}_weight.txt").reshape(DEFAULT_CHANNELS, -1)
        shift = self.ints("patch_embed_scalars.txt")[0]
        y = x.reshape(DEFAULT_TOKENS, -1) @ weight.T
        y = y + bias
        y[0, :] = cls
        return ((y + (1 << (shift - 1))) >> shift).reshape(-1)

    def mlp_block(self, index: int, input_values=None):
        prefix = f"mlp_{index}"
        results: list[GraphVerificationResult] = []
        x = self.tensor(f"{prefix}_input.txt") if input_values is None else input_values
        ln = self.layernorm(f"{prefix}_lnq", x)
        results.append(self.compare(f"{prefix}_lnq", "torch_int_layernorm", ln, f"{prefix}_lnq_output.txt"))
        m1 = self.static_matmul(f"{prefix}_matmul1", ln)
        results.append(self.compare(f"{prefix}_matmul1", "torch_int_matmul", m1, f"{prefix}_matmul1_output.txt"))
        ge = self.table_quant(f"{prefix}_geluq_scalars.txt", f"{prefix}_geluq_table_m.txt", m1)
        results.append(self.compare(f"{prefix}_geluq", "torch_int_gelu_table", ge, f"{prefix}_geluq_output.txt"))
        m2 = self.static_matmul(f"{prefix}_matmul2", ge)
        results.append(self.compare(f"{prefix}_matmul2", "torch_int_matmul", m2, f"{prefix}_matmul2_output.txt"))
        out = self.residual_merge(prefix, m2, x)
        results.append(self.compare(prefix, "torch_int_mlp_block", out, f"{prefix}_output.txt"))
        return out, results

    def attention_block(self, index: int, input_values=None):
        prefix = f"attn_{index}"
        results: list[GraphVerificationResult] = []
        x = self.tensor(f"{prefix}_input.txt") if input_values is None else input_values
        ln = self.layernorm(f"{prefix}_lnq", x)
        results.append(self.compare(f"{prefix}_lnq", "torch_int_layernorm", ln, f"{prefix}_lnq_output.txt"))

        q = self.static_matmul(f"{prefix}_gen_q_matmul", ln)
        k = self.static_matmul(f"{prefix}_gen_k_matmul", ln)
        v = self.static_matmul(f"{prefix}_gen_v_matmul", ln)
        results.append(self.compare(f"{prefix}_gen_q_matmul", "torch_int_matmul", q, f"{prefix}_gen_q_matmul_output.txt"))
        results.append(self.compare(f"{prefix}_gen_k_matmul", "torch_int_matmul", k, f"{prefix}_gen_k_matmul_output.txt"))
        results.append(self.compare(f"{prefix}_gen_v_matmul", "torch_int_matmul", v, f"{prefix}_gen_v_matmul_output.txt"))

        qq = self.table_quant(f"{prefix}_q_q_scalars.txt", f"{prefix}_q_q_table_m.txt", q)
        kq = self.table_quant(f"{prefix}_k_q_scalars.txt", f"{prefix}_k_q_table_m.txt", k)
        vq = self.table_quant(f"{prefix}_v_q_scalars.txt", f"{prefix}_v_q_table_m.txt", v)
        results.append(self.compare(f"{prefix}_qq", "torch_int_requant_table", qq, f"{prefix}_qq_output.txt"))
        results.append(self.compare(f"{prefix}_kq", "torch_int_requant_table", kq, f"{prefix}_kq_output.txt"))
        results.append(self.compare(f"{prefix}_vq", "torch_int_requant_table", vq, f"{prefix}_vq_output.txt"))

        q_heads = split_heads_tensor(qq, tokens=DEFAULT_TOKENS, channels=DEFAULT_CHANNELS, heads=DEFAULT_HEADS)
        k_heads = split_heads_tensor(kq, tokens=DEFAULT_TOKENS, channels=DEFAULT_CHANNELS, heads=DEFAULT_HEADS)
        v_heads_t = transpose_head_values_tensor(vq, tokens=DEFAULT_TOKENS, channels=DEFAULT_CHANNELS, heads=DEFAULT_HEADS)
        results.append(self.compare(f"{prefix}_gen_r_matmul_input", "torch_int_head_split", q_heads, f"{prefix}_gen_r_matmul_input.txt"))
        results.append(self.compare(f"{prefix}_gen_r_matmul_weight", "torch_int_head_split", k_heads, f"{prefix}_gen_r_matmul_weight.txt"))
        results.append(self.compare(f"{prefix}_gen_a_matmul_weight", "torch_int_head_transpose", v_heads_t, f"{prefix}_gen_a_matmul_weight.txt"))

        r = self.dynamic_head_matmul(f"{prefix}_gen_r_matmul", ci=64, co=DEFAULT_TOKENS, input_values=q_heads, weight_values=k_heads)
        results.append(self.compare(f"{prefix}_gen_r_matmul", "torch_int_dynamic_matmul", r, f"{prefix}_gen_r_matmul_output.txt"))
        softmax = self.softmax(f"{prefix}_softmaxq", r)
        results.append(self.compare(f"{prefix}_softmaxq", "torch_int_softmax_table", softmax, f"{prefix}_softmaxq_output.txt"))
        a = self.dynamic_head_matmul(f"{prefix}_gen_a_matmul", ci=DEFAULT_TOKENS, co=64, input_values=softmax, weight_values=v_heads_t)
        results.append(self.compare(f"{prefix}_gen_a_matmul", "torch_int_dynamic_matmul", a, f"{prefix}_gen_a_matmul_output.txt"))
        aq = self.table_quant(f"{prefix}_a_q_scalars.txt", f"{prefix}_a_q_table_m.txt", a)
        results.append(self.compare(f"{prefix}_aq", "torch_int_requant_table", aq, f"{prefix}_aq_output.txt"))
        o_input = merge_heads_tensor(aq, tokens=DEFAULT_TOKENS, channels=DEFAULT_CHANNELS, heads=DEFAULT_HEADS)
        results.append(self.compare(f"{prefix}_gen_o_matmul_input", "torch_int_head_merge", o_input, f"{prefix}_gen_o_matmul_input.txt"))
        o = self.static_matmul(f"{prefix}_gen_o_matmul", o_input)
        results.append(self.compare(f"{prefix}_gen_o_matmul", "torch_int_matmul", o, f"{prefix}_gen_o_matmul_output.txt"))
        out = self.residual_merge(prefix, o, x)
        results.append(self.compare(prefix, "torch_int_attention_block", out, f"{prefix}_output.txt"))
        return out, results

    def head(self, input_values=None):
        results: list[GraphVerificationResult] = []
        x = self.tensor("head_input.txt") if input_values is None else input_values
        cls = x.reshape(DEFAULT_TOKENS, DEFAULT_CHANNELS)[0]
        ln = self.layernorm("head_lnq", cls)
        results.append(self.compare("head_lnq", "torch_int_layernorm", ln, "head_lnq_output.txt"))
        logits = self.static_matmul("head_matmul", ln)
        results.append(self.compare("head_matmul", "torch_int_matmul", logits, "head_matmul_output.txt"))
        results.append(self.compare("head", "torch_int_head", logits, "head_output.txt"))
        return logits, results

    def forward_from_patch_input(self, input_values=None):
        results: list[GraphVerificationResult] = []
        current = self.patch_embed(input_values)
        results.append(self.compare("patch_embed", "torch_int_patch_embed", current, "patch_embed_output.txt"))
        results.append(self.compare("patch_embed_to_attn_0", "torch_int_chain_boundary", current, "attn_0_input.txt"))
        for index in range(12):
            current, attn_results = self.attention_block(index, current)
            results.extend(attn_results)
            results.append(self.compare(f"attn_{index}_to_mlp_{index}", "torch_int_chain_boundary", current, f"mlp_{index}_input.txt"))
            current, mlp_results = self.mlp_block(index, current)
            results.extend(mlp_results)
            next_file = f"attn_{index + 1}_input.txt" if index < 11 else "head_input.txt"
            next_name = f"mlp_{index}_to_attn_{index + 1}" if index < 11 else "mlp_11_to_head"
            results.append(self.compare(next_name, "torch_int_chain_boundary", current, next_file))
        logits, head_results = self.head(current)
        results.extend(head_results)
        return logits, results

    def verify_end_to_end(self) -> list[GraphVerificationResult]:
        _, results = self.forward_from_patch_input()
        return results

