from pathlib import Path

from hgpipe_quantization.artifacts import HgPipeSource
from hgpipe_quantization.pipeline import discover_cases, verify_case


def test_reference_sample_cases_are_bit_exact():
    source_path = Path(__file__).resolve().parents[2] / "ICCAD24-HG-PIPE"
    source = HgPipeSource.from_path(source_path)
    cases = {case.name: case for case in discover_cases(source)}
    for name in ["attn_0_qq", "attn_0_lnq", "attn_0_softmaxq", "mlp_0_geluq", "head_lnq"]:
        result = verify_case(cases[name])
        assert result.passed, result
