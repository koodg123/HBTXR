from hgpipe_quantization.ops import layernorm_quantize, quantize_clamp, softmax_quantize, table_quantize


def test_table_quantize_uses_hgpipe_cursor_formula():
    assert table_quantize([-2, 0, 2, 8], [2, 1, 3], [10, 11, 12, 13]) == [10, 11, 12, 13]


def test_quantize_clamp_signed_and_unsigned():
    assert quantize_clamp(-9, 4, signed=True) == -8
    assert quantize_clamp(8, 4, signed=True) == 7
    assert quantize_clamp(-1, 3, signed=False) == 0
    assert quantize_clamp(9, 3, signed=False) == 7


def test_layernorm_small_case_matches_hls_integer_steps():
    # C=2. mean=(sum*1 + round)/2, var table maps cursor directly to rsqrt=1.
    result = layernorm_quantize(
        inputs=[1, 3],
        scalars=[1, 1, 0, 0, 10, 0, 4],
        lnw=[1, 1],
        lnb=[0, 0],
        rsqrt_table=[1] * 11,
    )
    assert result == [-1, 1]


def test_softmax_small_case_reconstructs_segmented_tables():
    result = softmax_quantize(
        inputs=[1, 0, 0, 1],
        scalars=[0, 0, 3, 0, 0, 10, 0, 0, 0, 0, 10, 0, 0, 3],
        exp_table=[4, 2, 1, 0],
        recip_table_one=[0, 1, 2, 3, 4, 5, 6],
        recip_table_two=[0],
        tokens=2,
        heads=1,
    )
    assert result == [7, 7, 7, 7]
