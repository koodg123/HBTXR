#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import math
import re
from pathlib import Path

def gelu_ref(x: float) -> float:
    return 0.5 * x * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))

DEFAULT_CONTRACT = Path("refs/hgpipe_lut_math_contract.json")
GELUQ_KEYS = [f"gelu_quantized_mlp{layer}" for layer in range(12)]
ATTN_QUANT_KEYS = [
    f"quant_attn{layer}_{kind}"
    for layer in range(12)
    for kind in ("q", "k", "v", "a")
]
SOFTMAX_KEYS = [f"softmax_attn{layer}" for layer in range(12)]
LAYERNORM_KEYS = [
    *(f"layernorm_attn{layer}" for layer in range(12)),
    *(f"layernorm_mlp{layer}" for layer in range(12)),
    "layernorm_head",
]

def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))

def load_contract(path: Path) -> dict:
    contract = json.loads(path.read_text())
    if contract.get("schema") != "hgtxr.hgpipe_lut_math_contract.v1":
        raise ValueError(f"unsupported contract schema in {path}")
    return contract

def require_len(name: str, table: list[float], expected: int) -> None:
    if len(table) != expected:
        raise ValueError(f"{name} table has {len(table)} entries, expected {expected}")

def parse_int_csv(path: Path) -> list[int]:
    return [int(x) for x in re.findall(r"-?\d+", path.read_text())]

def ap_int_range(type_name: str) -> tuple[int, int]:
    match = re.fullmatch(r"ap_(u?)int<(\d+)>", type_name.replace(" ", ""))
    if not match:
        raise ValueError(f"unsupported integer type {type_name}")
    unsigned, bits_text = match.groups()
    bits = int(bits_text)
    if unsigned:
        return 0, (1 << bits) - 1
    return -(1 << (bits - 1)), (1 << (bits - 1)) - 1

def require_values_fit(name: str, values: list[int], type_name: str) -> None:
    lo, hi = ap_int_range(type_name)
    actual_lo = min(values)
    actual_hi = max(values)
    if actual_lo < lo or actual_hi > hi:
        raise ValueError(f"{name} range [{actual_lo}, {actual_hi}] does not fit {type_name} [{lo}, {hi}]")

def gelu_lut(table: list[float], x: float) -> float:
    return table[clamp(int((x + 4.0) * 1.875), 0, 15)]

def exp_lut(table: list[float], shifted_score: float) -> float:
    opposite = -shifted_score if shifted_score < 0.0 else 0.0
    return table[clamp(int(opposite * 1.875), 0, 15)]

def rsqrt_lut(table: list[float], var_mean: float) -> float:
    clipped = min(64.0, max(1.0 / 128.0, var_mean))
    return table[clamp(int(clipped * 0.484375), 0, 31)]

def max_error(samples: list[float], lut_fn, ref_fn) -> dict[str, float]:
    errors = [abs(lut_fn(x) - ref_fn(x)) for x in samples]
    return {"max_abs_error": max(errors), "mean_abs_error": sum(errors) / len(errors)}

def quantize_clamp(value: int, bits: int, signed_output: bool) -> int:
    if signed_output:
        lo = -(1 << (bits - 1))
        hi = (1 << (bits - 1)) - 1
    else:
        lo = 0
        hi = (1 << bits) - 1
    return clamp(value, lo, hi)

def hgpipe_cursor_lut(value: int, b: int, s: int, bound: int, table: list[int]) -> tuple[int, int]:
    cursor = clamp((value + b) >> s, 0, bound)
    return cursor, table[cursor]

def validate_cursor_table_refs(contract: dict, key: str, table: list[int]) -> dict:
    refs = contract["hgpipe_ref_files"]
    cursor = contract["cursor_contracts"][key]
    scalars = cursor["scalars"]
    b = int(scalars["b"])
    s = int(scalars["s"])
    bound = int(scalars["bound"])

    scalar_ref = parse_int_csv(Path(refs[f"{key}_scalars"]))
    table_ref = parse_int_csv(Path(refs[f"{key}_table"]))
    input_ref = parse_int_csv(Path(refs[f"{key}_input"]))
    output_ref = parse_int_csv(Path(refs[f"{key}_output"]))
    expected_samples = math.prod(int(x) for x in cursor["sample_shape"])

    if scalar_ref != [b, s, bound]:
        raise ValueError(f"{key} scalar mismatch: {scalar_ref}")
    if table_ref != table:
        raise ValueError(f"{key} table mismatch against HG-PIPE ref")
    if len(input_ref) != expected_samples:
        raise ValueError(f"{key} input count {len(input_ref)}, expected {expected_samples}")
    if len(output_ref) != expected_samples:
        raise ValueError(f"{key} output count {len(output_ref)}, expected {expected_samples}")
    if bound < 0 or bound >= len(table):
        raise ValueError(f"{key} bound {bound} outside table entries {len(table)}")

    require_values_fit(f"{key} input", input_ref, cursor["input_type"])
    require_values_fit(f"{key} table", table_ref, cursor["output_type"])
    require_values_fit(f"{key} output", output_ref, cursor["output_type"])

    mismatches = []
    cursor_min = bound
    cursor_max = 0
    raw_cursor_min = None
    raw_cursor_max = None
    clamp_low_hits = 0
    clamp_high_hits = 0
    output_min = table[0]
    output_max = table[0]
    for idx, (x, expected) in enumerate(zip(input_ref, output_ref)):
        raw_cursor = (x + b) >> s
        cursor_idx, actual = hgpipe_cursor_lut(x, b, s, bound, table)
        raw_cursor_min = raw_cursor if raw_cursor_min is None else min(raw_cursor_min, raw_cursor)
        raw_cursor_max = raw_cursor if raw_cursor_max is None else max(raw_cursor_max, raw_cursor)
        if raw_cursor < 0:
            clamp_low_hits += 1
        if raw_cursor > bound:
            clamp_high_hits += 1
        cursor_min = min(cursor_min, cursor_idx)
        cursor_max = max(cursor_max, cursor_idx)
        output_min = min(output_min, actual)
        output_max = max(output_max, actual)
        if actual != expected:
            mismatches.append({"index": idx, "input": x, "cursor": cursor_idx, "actual": actual, "expected": expected})
            if len(mismatches) >= 8:
                break

    if mismatches:
        raise ValueError(f"{key} mismatches: {mismatches}")

    return {
        "status": "pass",
        "checked_samples": expected_samples,
        "scalars": {"b": b, "s": s, "bound": bound},
        "table_entries": len(table),
        "input_range": [min(input_ref), max(input_ref)],
        "raw_cursor_range": [raw_cursor_min, raw_cursor_max],
        "cursor_range": [cursor_min, cursor_max],
        "clamp_hits": {"low": clamp_low_hits, "high": clamp_high_hits},
        "output_range": [output_min, output_max],
        "mismatches": 0,
    }

def validate_softmax_refs(contract: dict,
                          key: str,
                          exp_table: list[int],
                          recip_table_one: list[int],
                          recip_table_two: list[int]) -> dict:
    refs = contract["hgpipe_ref_files"]
    cursor = contract["cursor_contracts"][key]
    scalars = cursor["scalars"]
    scalar_order = [
        "b1", "s1", "bound1", "b2_one", "s2_one", "bound2_one", "b3_one", "s3_one",
        "b2_two", "s2_two", "bound2_two", "b3_two", "s3_two", "clamp_bits",
    ]
    scalar_ref = parse_int_csv(Path(refs[f"{key}_scalars"]))
    expected_scalars = [int(scalars[name]) for name in scalar_order]
    if scalar_ref != expected_scalars:
        raise ValueError(f"{key} scalar mismatch: {scalar_ref}")

    exp_ref = parse_int_csv(Path(refs[f"{key}_exp_table"]))
    recip_one_ref = parse_int_csv(Path(refs[f"{key}_recip_table_one"]))
    recip_two_ref = parse_int_csv(Path(refs[f"{key}_recip_table_two"]))
    input_ref = parse_int_csv(Path(refs[f"{key}_input"]))
    output_ref = parse_int_csv(Path(refs[f"{key}_output"]))

    if exp_ref != exp_table:
        raise ValueError(f"{key} exp table mismatch against HG-PIPE ref")
    if recip_one_ref != recip_table_one:
        raise ValueError(f"{key} recip table one mismatch against HG-PIPE ref")
    if recip_two_ref != recip_table_two:
        raise ValueError(f"{key} recip table two mismatch against HG-PIPE ref")

    heads, rows, cols = [int(x) for x in cursor["sample_shape"]]
    expected_samples = heads * rows * cols
    if len(input_ref) != expected_samples:
        raise ValueError(f"{key} input count {len(input_ref)}, expected {expected_samples}")
    if len(output_ref) != expected_samples:
        raise ValueError(f"{key} output count {len(output_ref)}, expected {expected_samples}")

    require_values_fit(f"{key} input", input_ref, cursor["input_type"])
    require_values_fit(f"{key} exp_table", exp_table, cursor["exp_type"])
    require_values_fit(f"{key} recip_table_one", recip_table_one, cursor["recip_type"])
    require_values_fit(f"{key} recip_table_two", recip_table_two, cursor["recip_type"])
    require_values_fit(f"{key} output", output_ref, cursor["output_type"])

    b1 = int(scalars["b1"])
    s1 = int(scalars["s1"])
    bound1 = int(scalars["bound1"])
    b2_one = int(scalars["b2_one"])
    s2_one = int(scalars["s2_one"])
    bound2_one = int(scalars["bound2_one"])
    b3_one = int(scalars["b3_one"])
    s3_one = int(scalars["s3_one"])
    b2_two = int(scalars["b2_two"])
    s2_two = int(scalars["s2_two"])
    bound2_two = int(scalars["bound2_two"])
    b3_two = int(scalars["b3_two"])
    s3_two = int(scalars["s3_two"])
    clamp_bits = int(scalars["clamp_bits"])

    if bound1 < 0 or bound1 >= len(exp_table):
        raise ValueError(f"{key} bound1 {bound1} outside exp table")
    if bound2_one < 0 or bound2_one >= len(recip_table_one):
        raise ValueError(f"{key} bound2_one {bound2_one} outside recip table one")
    if bound2_two < 0 or bound2_two >= len(recip_table_two):
        raise ValueError(f"{key} bound2_two {bound2_two} outside recip table two")

    exp_cursor_min = bound1
    exp_cursor_max = 0
    exp_clamp_low = 0
    exp_clamp_high = 0
    acc_min = None
    acc_max = None
    branch_two_count = 0
    recip_one_cursor_min = bound2_one
    recip_one_cursor_max = 0
    recip_two_cursor_min = bound2_two
    recip_two_cursor_max = 0
    output_min = (1 << clamp_bits) - 1
    output_max = 0
    mismatches = []

    for h in range(heads):
        for row in range(rows):
            offset = (h * rows + row) * cols
            row_values = input_ref[offset:offset + cols]
            expected_values = output_ref[offset:offset + cols]
            max_value = max(row_values)
            exp_scores = []
            acc = 0
            for value in row_values:
                opposite_delta = max_value - value
                raw_exp_cursor = (opposite_delta + b1) >> s1
                if raw_exp_cursor < 0:
                    exp_clamp_low += 1
                if raw_exp_cursor > bound1:
                    exp_clamp_high += 1
                exp_cursor = clamp(raw_exp_cursor, 0, bound1)
                exp_cursor_min = min(exp_cursor_min, exp_cursor)
                exp_cursor_max = max(exp_cursor_max, exp_cursor)
                exp_value = exp_table[exp_cursor]
                exp_scores.append(exp_value)
                acc += exp_value

            acc_min = acc if acc_min is None else min(acc_min, acc)
            acc_max = acc if acc_max is None else max(acc_max, acc)
            recip_one_raw = (acc + b2_one) >> s2_one
            if recip_one_raw > bound2_one:
                branch_two_count += 1
                recip_two_cursor = clamp((acc + b2_two) >> s2_two, 0, bound2_two)
                recip_two_cursor_min = min(recip_two_cursor_min, recip_two_cursor)
                recip_two_cursor_max = max(recip_two_cursor_max, recip_two_cursor)
                recip = recip_table_two[recip_two_cursor]
                rel_b = b3_two
                rel_s = s3_two
            else:
                recip_one_cursor = clamp(recip_one_raw, 0, bound2_one)
                recip_one_cursor_min = min(recip_one_cursor_min, recip_one_cursor)
                recip_one_cursor_max = max(recip_one_cursor_max, recip_one_cursor)
                recip = recip_table_one[recip_one_cursor]
                rel_b = b3_one
                rel_s = s3_one

            for col, (exp_value, expected) in enumerate(zip(exp_scores, expected_values)):
                rel = ((exp_value * recip) + rel_b) >> rel_s
                actual = quantize_clamp(rel, clamp_bits, False)
                output_min = min(output_min, actual)
                output_max = max(output_max, actual)
                if actual != expected:
                    mismatches.append({
                        "head": h,
                        "row": row,
                        "col": col,
                        "input": row_values[col],
                        "actual": actual,
                        "expected": expected,
                        "acc": acc,
                    })
                    if len(mismatches) >= 8:
                        break
            if mismatches:
                break
        if mismatches:
            break

    if mismatches:
        raise ValueError(f"{key} mismatches: {mismatches}")

    return {
        "status": "pass",
        "checked_samples": expected_samples,
        "scalars": {name: int(scalars[name]) for name in scalar_order},
        "table_entries": {
            "exp": len(exp_table),
            "recip_one": len(recip_table_one),
            "recip_two": len(recip_table_two),
        },
        "input_range": [min(input_ref), max(input_ref)],
        "exp_cursor_range": [exp_cursor_min, exp_cursor_max],
        "exp_clamp_hits": {"low": exp_clamp_low, "high": exp_clamp_high},
        "acc_range": [acc_min, acc_max],
        "recip_branch_two_rows": branch_two_count,
        "recip_one_cursor_range": [recip_one_cursor_min, recip_one_cursor_max],
        "recip_two_cursor_range": [recip_two_cursor_min, recip_two_cursor_max],
        "output_range": [output_min, output_max],
        "mismatches": 0,
    }

def validate_layernorm_refs(contract: dict, key: str, rsqrt_table: list[int]) -> dict:
    refs = contract["hgpipe_ref_files"]
    cursor = contract["cursor_contracts"][key]
    scalars = cursor["scalars"]
    scalar_order = ["C_1_m", "C_1_s", "b", "s1", "bound", "s2", "clamp_bits"]
    scalar_ref = parse_int_csv(Path(refs[f"{key}_scalars"]))
    expected_scalars = [int(scalars[name]) for name in scalar_order]
    if scalar_ref != expected_scalars:
        raise ValueError(f"{key} scalar mismatch: {scalar_ref}")

    rsqrt_ref = parse_int_csv(Path(refs[f"{key}_rsqrt_table"]))
    lnw_ref = parse_int_csv(Path(refs[f"{key}_lnw"]))
    lnb_ref = parse_int_csv(Path(refs[f"{key}_lnb"]))
    input_ref = parse_int_csv(Path(refs[f"{key}_input"]))
    output_ref = parse_int_csv(Path(refs[f"{key}_output"]))

    if rsqrt_ref != rsqrt_table:
        raise ValueError(f"{key} rsqrt table mismatch against HG-PIPE ref")

    rows, cols = [int(x) for x in cursor["sample_shape"]]
    expected_samples = rows * cols
    if len(lnw_ref) != cols:
        raise ValueError(f"{key} lnw count {len(lnw_ref)}, expected {cols}")
    if len(lnb_ref) != cols:
        raise ValueError(f"{key} lnb count {len(lnb_ref)}, expected {cols}")
    if len(input_ref) != expected_samples:
        raise ValueError(f"{key} input count {len(input_ref)}, expected {expected_samples}")
    if len(output_ref) != expected_samples:
        raise ValueError(f"{key} output count {len(output_ref)}, expected {expected_samples}")

    require_values_fit(f"{key} input", input_ref, cursor["input_type"])
    require_values_fit(f"{key} mean", [0], cursor["mean_type"])
    require_values_fit(f"{key} variance_sum", [0], cursor["variance_sum_type"])
    require_values_fit(f"{key} output", output_ref, cursor["output_type"])

    c_1_m = int(scalars["C_1_m"])
    c_1_s = int(scalars["C_1_s"])
    b = int(scalars["b"])
    s1 = int(scalars["s1"])
    bound = int(scalars["bound"])
    s2 = int(scalars["s2"])
    clamp_bits = int(scalars["clamp_bits"])
    if bound < 0 or bound >= len(rsqrt_table):
        raise ValueError(f"{key} bound {bound} outside rsqrt table")

    mean_min = None
    mean_max = None
    var_min = None
    var_max = None
    raw_cursor_min = None
    raw_cursor_max = None
    cursor_min = bound
    cursor_max = 0
    clamp_low_hits = 0
    clamp_high_hits = 0
    rel_min = None
    rel_max = None
    output_min = None
    output_max = None
    rsqrt_used_min = None
    rsqrt_used_max = None
    mismatches = []

    for row in range(rows):
        offset = row * cols
        values = input_ref[offset:offset + cols]
        expected_values = output_ref[offset:offset + cols]
        acc = sum(values)
        mean = ((acc * c_1_m) + (1 << (c_1_s - 1))) >> c_1_s
        variance_sum = sum((x - mean) * (x - mean) for x in values)
        raw_cursor = (variance_sum + b) >> s1
        if raw_cursor < 0:
            clamp_low_hits += 1
        if raw_cursor > bound:
            clamp_high_hits += 1
        cursor_idx = clamp(raw_cursor, 0, bound)
        rsqrt = rsqrt_table[cursor_idx]
        rsqrt_used_min = rsqrt if rsqrt_used_min is None else min(rsqrt_used_min, rsqrt)
        rsqrt_used_max = rsqrt if rsqrt_used_max is None else max(rsqrt_used_max, rsqrt)

        mean_min = mean if mean_min is None else min(mean_min, mean)
        mean_max = mean if mean_max is None else max(mean_max, mean)
        var_min = variance_sum if var_min is None else min(var_min, variance_sum)
        var_max = variance_sum if var_max is None else max(var_max, variance_sum)
        raw_cursor_min = raw_cursor if raw_cursor_min is None else min(raw_cursor_min, raw_cursor)
        raw_cursor_max = raw_cursor if raw_cursor_max is None else max(raw_cursor_max, raw_cursor)
        cursor_min = min(cursor_min, cursor_idx)
        cursor_max = max(cursor_max, cursor_idx)

        for col, (value, expected) in enumerate(zip(values, expected_values)):
            val = (value - mean) * rsqrt * lnw_ref[col] + lnb_ref[col]
            rel = val >> s2
            actual = quantize_clamp(rel, clamp_bits, True)
            rel_min = rel if rel_min is None else min(rel_min, rel)
            rel_max = rel if rel_max is None else max(rel_max, rel)
            output_min = actual if output_min is None else min(output_min, actual)
            output_max = actual if output_max is None else max(output_max, actual)
            if actual != expected:
                mismatches.append({
                    "row": row,
                    "col": col,
                    "input": value,
                    "mean": mean,
                    "variance_sum": variance_sum,
                    "cursor": cursor_idx,
                    "actual": actual,
                    "expected": expected,
                })
                if len(mismatches) >= 8:
                    break
        if mismatches:
            break

    if mismatches:
        raise ValueError(f"{key} mismatches: {mismatches}")

    require_values_fit(f"{key} mean observed", [mean_min, mean_max], cursor["mean_type"])
    require_values_fit(f"{key} variance observed", [var_min, var_max], cursor["variance_sum_type"])
    require_values_fit(f"{key} cursor observed", [cursor_min, cursor_max], cursor["cursor_type"])
    require_values_fit(f"{key} rsqrt observed", [rsqrt_used_min, rsqrt_used_max], cursor["rsqrt_type"])

    return {
        "status": "pass",
        "checked_samples": expected_samples,
        "scalars": {name: int(scalars[name]) for name in scalar_order},
        "table_entries": {"rsqrt": len(rsqrt_table), "lnw": len(lnw_ref), "lnb": len(lnb_ref)},
        "input_range": [min(input_ref), max(input_ref)],
        "mean_range": [mean_min, mean_max],
        "variance_sum_range": [var_min, var_max],
        "raw_cursor_range": [raw_cursor_min, raw_cursor_max],
        "cursor_range": [cursor_min, cursor_max],
        "clamp_hits": {"low": clamp_low_hits, "high": clamp_high_hits},
        "rsqrt_range": [min(rsqrt_table), max(rsqrt_table)],
        "rsqrt_used_range": [rsqrt_used_min, rsqrt_used_max],
        "lnw_range": [min(lnw_ref), max(lnw_ref)],
        "lnb_range": [min(lnb_ref), max(lnb_ref)],
        "requant_shift_range": [rel_min, rel_max],
        "output_range": [output_min, output_max],
        "mismatches": 0,
    }

def validate_contract(contract: dict) -> dict:
    tables = contract["tables"]
    cursor = contract["cursor_contracts"]
    gelu_table = [float(x) for x in tables["gelu"]]
    geluq_tables = {key: [int(x) for x in tables[key]] for key in GELUQ_KEYS}
    attn_quant_tables = {key: [int(x) for x in tables[key]] for key in ATTN_QUANT_KEYS}
    softmax_tables = {
        key: {
            "exp": [int(x) for x in tables[f"{key}_exp"]],
            "recip_one": [int(x) for x in tables[f"{key}_recip_one"]],
            "recip_two": [int(x) for x in tables[f"{key}_recip_two"]],
        }
        for key in SOFTMAX_KEYS
    }
    layernorm_rsqrt_tables = {key: [int(x) for x in tables[f"{key}_rsqrt"]] for key in LAYERNORM_KEYS}
    exp_table = [float(x) for x in tables["softmax_exp"]]
    rsqrt_table = [float(x) for x in tables["layernorm_rsqrt"]]
    require_len("gelu", gelu_table, int(cursor["gelu"]["entries"]))
    for key, table in geluq_tables.items():
        require_len(key, table, int(cursor[key]["entries"]))
    for key, table in attn_quant_tables.items():
        require_len(key, table, int(cursor[key]["entries"]))
    for key, softmax_table in softmax_tables.items():
        require_len(f"{key}_exp", softmax_table["exp"], int(cursor[key]["entries_exp"]))
        require_len(f"{key}_recip_one", softmax_table["recip_one"], int(cursor[key]["entries_recip"]))
        require_len(f"{key}_recip_two", softmax_table["recip_two"], int(cursor[key]["entries_recip"]))
    for key, table in layernorm_rsqrt_tables.items():
        require_len(f"{key}_rsqrt", table, int(cursor[key]["entries_rsqrt"]))
    require_len("softmax_exp", exp_table, int(cursor["softmax_exp"]["entries"]))
    require_len("layernorm_rsqrt", rsqrt_table, int(cursor["layernorm_rsqrt"]["entries"]))

    quant = cursor["quantize_clamp"]
    signed_bits = int(quant["signed_bits"])
    unsigned_bits = int(quant["unsigned_bits"])
    signed_range = [quantize_clamp(v, signed_bits, True) for v in (-999, -128, -1, 0, 127, 999)]
    unsigned_range = [quantize_clamp(v, unsigned_bits, False) for v in (-999, -1, 0, 1, 255, 999)]
    if signed_range != [-128, -128, -1, 0, 127, 127]:
        raise ValueError(f"signed quant clamp mismatch: {signed_range}")
    if unsigned_range != [0, 0, 0, 1, 255, 255]:
        raise ValueError(f"unsigned quant clamp mismatch: {unsigned_range}")

    return {
        "gelu_table": gelu_table,
        "geluq_tables": geluq_tables,
        "attn_quant_tables": attn_quant_tables,
        "softmax_tables": softmax_tables,
        "layernorm_rsqrt_tables": layernorm_rsqrt_tables,
        "exp_table": exp_table,
        "rsqrt_table": rsqrt_table,
        "quant_clamp_examples": {
            "signed_8": signed_range,
            "unsigned_8": unsigned_range,
        },
        "hgpipe_ref_checks": {
            **{key: validate_cursor_table_refs(contract, key, table) for key, table in geluq_tables.items()},
            **{key: validate_cursor_table_refs(contract, key, table) for key, table in attn_quant_tables.items()},
            **{
                key: validate_softmax_refs(
                    contract,
                    key,
                    table["exp"],
                    table["recip_one"],
                    table["recip_two"],
                )
                for key, table in softmax_tables.items()
            },
            **{key: validate_layernorm_refs(contract, key, table) for key, table in layernorm_rsqrt_tables.items()},
        },
    }

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--out", type=Path, default=Path("docs/resources/hgpipe_lut_math_validation_2026_06_09.json"))
    args = parser.parse_args()
    contract = load_contract(args.contract)
    validated = validate_contract(contract)
    gelu_table = validated["gelu_table"]
    geluq_tables = validated["geluq_tables"]
    attn_quant_tables = validated["attn_quant_tables"]
    softmax_tables = validated["softmax_tables"]
    layernorm_rsqrt_tables = validated["layernorm_rsqrt_tables"]
    exp_table = validated["exp_table"]
    rsqrt_table = validated["rsqrt_table"]
    gelu_samples = [-4.0 + 8.0 * i / 512.0 for i in range(513)]
    exp_samples = [-8.0 + 8.0 * i / 512.0 for i in range(513)]
    rsqrt_samples = [1.0 / 128.0 + (64.0 - 1.0 / 128.0) * i / 512.0 for i in range(513)]
    report = {
        "status": "pass",
        "contract": str(args.contract),
        "source_pattern": "HG-PIPE cursor plus clamp plus LUT for GeLU, Softmax exp/recip family, LayerNorm rsqrt, and Quant clamp",
        "hgpipe_sources": contract["hgpipe_sources"],
        "hgpipe_ref_files": contract["hgpipe_ref_files"],
        "hgtxr_sources": contract["hgtxr_sources"],
        "cursor_contracts": contract["cursor_contracts"],
        "tables": {
            "gelu_entries": len(gelu_table),
            **{f"{key}_entries": len(table) for key, table in geluq_tables.items()},
            **{f"{key}_entries": len(table) for key, table in attn_quant_tables.items()},
            **{f"{key}_exp_entries": len(table["exp"]) for key, table in softmax_tables.items()},
            **{f"{key}_recip_one_entries": len(table["recip_one"]) for key, table in softmax_tables.items()},
            **{f"{key}_recip_two_entries": len(table["recip_two"]) for key, table in softmax_tables.items()},
            **{f"{key}_rsqrt_entries": len(table) for key, table in layernorm_rsqrt_tables.items()},
            "exp_entries": len(exp_table),
            "rsqrt_entries": len(rsqrt_table),
        },
        "quant_clamp_examples": validated["quant_clamp_examples"],
        "hgpipe_ref_checks": validated["hgpipe_ref_checks"],
        "error_summary": {
            "gelu_x_range_neg4_pos4": max_error(gelu_samples, lambda x: gelu_lut(gelu_table, x), gelu_ref),
            "exp_shifted_range_neg8_0": max_error(exp_samples, lambda x: exp_lut(exp_table, x), math.exp),
            "rsqrt_var_range_0p0078125_64": max_error(rsqrt_samples, lambda x: rsqrt_lut(rsqrt_table, x), lambda x: 1.0 / math.sqrt(x)),
        },
        "acceptance_note": contract["acceptance"]["purpose"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + chr(10))
    print(json.dumps(report["error_summary"], indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
