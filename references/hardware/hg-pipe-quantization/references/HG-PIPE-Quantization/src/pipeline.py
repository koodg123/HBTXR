"""End-to-end discovery and verification for HG-PIPE quantization artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .artifacts import HgPipeSource, load_statistics, metadata_for, read_ints, summarize_tensor
from .ops import layernorm_quantize, softmax_quantize, table_quantize


@dataclass(frozen=True)
class QuantizationCase:
    name: str
    kind: str
    input_path: Path
    output_path: Path
    scalars_path: Path
    table_paths: tuple[Path, ...]


@dataclass(frozen=True)
class VerificationResult:
    name: str
    kind: str
    passed: bool
    elements: int
    mismatches: int
    metadata: dict[str, Any]


def _exists(*paths: Path) -> bool:
    return all(path.exists() for path in paths)


def discover_cases(source: HgPipeSource | str | Path) -> list[QuantizationCase]:
    if not isinstance(source, HgPipeSource):
        source = HgPipeSource.from_path(source)

    refs = source.refs
    cases: list[QuantizationCase] = []

    for scalars in sorted(refs.glob("*_q_scalars.txt")):
        stem = scalars.name[: -len("_q_scalars.txt")]
        table = refs / f"{stem}_q_table_m.txt"
        io_stem = f"{stem}q"
        input_path = refs / f"{io_stem}_input.txt"
        output_path = refs / f"{io_stem}_output.txt"
        if _exists(table, input_path, output_path):
            cases.append(QuantizationCase(io_stem, "requant_table", input_path, output_path, scalars, (table,)))

    for scalars in sorted(refs.glob("*_geluq_scalars.txt")):
        io_stem = scalars.name[: -len("_scalars.txt")]
        table = refs / f"{io_stem}_table_m.txt"
        input_path = refs / f"{io_stem}_input.txt"
        output_path = refs / f"{io_stem}_output.txt"
        if _exists(table, input_path, output_path):
            cases.append(QuantizationCase(io_stem, "gelu_requant_table", input_path, output_path, scalars, (table,)))

    for scalars in sorted(refs.glob("*_lnq_scalars.txt")):
        io_stem = scalars.name[: -len("_scalars.txt")]
        lnw = refs / f"{io_stem}_lnw_m.txt"
        lnb = refs / f"{io_stem}_lnb_m.txt"
        rsqrt = refs / f"{io_stem}_rsqrt_table_m.txt"
        input_path = refs / f"{io_stem}_input.txt"
        output_path = refs / f"{io_stem}_output.txt"
        if _exists(lnw, lnb, rsqrt, input_path, output_path):
            cases.append(QuantizationCase(io_stem, "layernorm_rsqrt_table", input_path, output_path, scalars, (lnw, lnb, rsqrt)))

    for scalars in sorted(refs.glob("*_softmaxq_scalars.txt")):
        io_stem = scalars.name[: -len("_scalars.txt")]
        exp_table = refs / f"{io_stem}_exp_opp_table_m.txt"
        recip_one = refs / f"{io_stem}_recip_scaled_table_m_one.txt"
        recip_two = refs / f"{io_stem}_recip_scaled_table_m_two.txt"
        input_path = refs / f"{io_stem}_input.txt"
        output_path = refs / f"{io_stem}_output.txt"
        if _exists(exp_table, recip_one, recip_two, input_path, output_path):
            cases.append(
                QuantizationCase(
                    io_stem,
                    "softmax_segmented_table",
                    input_path,
                    output_path,
                    scalars,
                    (exp_table, recip_one, recip_two),
                )
            )

    return cases


def _kernel_for(case: QuantizationCase) -> Callable[[], list[int]]:
    inputs = read_ints(case.input_path)
    scalars = read_ints(case.scalars_path)
    tables = [read_ints(path) for path in case.table_paths]

    if case.kind in {"requant_table", "gelu_requant_table"}:
        return lambda: table_quantize(inputs, scalars, tables[0])
    if case.kind == "layernorm_rsqrt_table":
        return lambda: layernorm_quantize(inputs, scalars, tables[0], tables[1], tables[2])
    if case.kind == "softmax_segmented_table":
        return lambda: softmax_quantize(inputs, scalars, tables[0], tables[1], tables[2])
    raise ValueError(f"Unknown quantization case kind: {case.kind}")


def verify_case(case: QuantizationCase, stats: dict[str, Any] | None = None) -> VerificationResult:
    expected = read_ints(case.output_path)
    actual = _kernel_for(case)()
    mismatches = sum(1 for got, want in zip(actual, expected) if got != want)
    if len(actual) != len(expected):
        mismatches += abs(len(actual) - len(expected))

    metadata = metadata_for(case.name, stats or {})
    metadata["input_summary"] = summarize_tensor(read_ints(case.input_path))
    metadata["output_summary"] = summarize_tensor(expected)
    metadata["scalars"] = read_ints(case.scalars_path)
    metadata["table_sizes"] = [len(read_ints(path)) for path in case.table_paths]

    return VerificationResult(
        name=case.name,
        kind=case.kind,
        passed=mismatches == 0,
        elements=len(expected),
        mismatches=mismatches,
        metadata=metadata,
    )


def verify_all(source: HgPipeSource | str | Path) -> list[VerificationResult]:
    if not isinstance(source, HgPipeSource):
        source = HgPipeSource.from_path(source)
    stats = load_statistics(source)
    return [verify_case(case, stats) for case in discover_cases(source)]
