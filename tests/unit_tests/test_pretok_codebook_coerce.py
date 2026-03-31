from torchtitan.datasets.hf_datasets import _coerce_mimi_codes


def test_coerce_tq_truncates_higher_q_to_requested_q():
    # [T, Q=8] input should support requested Q=5 by truncating each frame.
    raw = [
        [10, 11, 12, 13, 14, 15, 16, 17],
        [20, 21, 22, 23, 24, 25, 26, 27],
    ]
    out = _coerce_mimi_codes(raw, num_quantizers=5)
    assert out == [
        [10, 11, 12, 13, 14],
        [20, 21, 22, 23, 24],
    ]


def test_coerce_qt_truncates_higher_q_to_requested_q():
    # [Q=8, T] layout should also support requested Q=6.
    raw = [
        [100, 101, 102],  # q0
        [110, 111, 112],  # q1
        [120, 121, 122],  # q2
        [130, 131, 132],  # q3
        [140, 141, 142],  # q4
        [150, 151, 152],  # q5
        [160, 161, 162],  # q6
        [170, 171, 172],  # q7
    ]
    out = _coerce_mimi_codes(raw, num_quantizers=6)
    assert out == [
        [100, 110, 120, 130, 140, 150],
        [101, 111, 121, 131, 141, 151],
        [102, 112, 122, 132, 142, 152],
    ]


def test_coerce_single_frame_tq_is_preserved():
    # Valid [T=1, Q=8] should not be mistaken for [Q, T].
    raw = [[1, 2, 3, 4, 5, 6, 7, 8]]
    out = _coerce_mimi_codes(raw, num_quantizers=5)
    assert out == [[1, 2, 3, 4, 5]]
