from pathlib import Path

from hgpipe_quantization.quant_params import QuantParamStore


def test_quant_param_store_loads_lut_contract_and_statistics():
    source_path = Path(__file__).resolve().parents[2] / "ICCAD24-HG-PIPE"
    store = QuantParamStore(source_path)

    contract = store.contract_for_table_case(
        "attn_0_qq",
        "requant_table",
        "attn_0_q_q_scalars.txt",
        "attn_0_q_q_table_m.txt",
    )

    assert contract.stat_key == "attn0.qq"
    assert contract.params is not None
    assert len(contract.params.scalars) == 3
    assert len(contract.params.tables) == 1
    assert contract.output_dtype is not None
    assert contract.output_dtype.bits == 3
    assert contract.output_dtype.signed


def test_quant_param_store_exposes_patch_embed_input_contract():
    source_path = Path(__file__).resolve().parents[2] / "ICCAD24-HG-PIPE"
    store = QuantParamStore(source_path)

    dtype = store.tensor_dtype("patch_embed.input")
    observed = store.observed_range_group("patch_embed")

    assert dtype is not None
    assert dtype.signed
    assert dtype.bits == 8
    assert observed["input"].minimum == -102.0
    assert observed["input"].maximum == 127.0

