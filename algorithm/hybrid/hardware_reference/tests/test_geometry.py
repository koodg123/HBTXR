from software.geometry import compute_open_extent_from_binary_mask, ellipse_mask


def test_ellipse_mask_open_extent():
    mask = ellipse_mask((64, 64), [32, 32, 20, 10, 0])
    assert mask.sum() > 0
    assert compute_open_extent_from_binary_mask(mask, [32, 32, 20, 10, 0]) > 0.9

