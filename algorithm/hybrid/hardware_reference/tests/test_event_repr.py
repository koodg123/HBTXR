import numpy as np

from software.event_repr import to_polarity_frame


def test_to_polarity_frame():
    events = np.array([[1, 2, 0, 1], [1, 2, 1, -1]], dtype=np.float32)
    frame = to_polarity_frame(events, height=8, width=8)
    assert frame.shape == (2, 8, 8)
    assert frame[:, 2, 1].sum() > 0

