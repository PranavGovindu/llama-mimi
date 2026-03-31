import torch

from torchtitan.tools.audio_token_parser import (
    build_audio_code_id_map,
    build_spark_global_id_map,
    extract_audio_codes_bqt_from_token_ids,
    extract_spark_global_token_ids,
    filter_tokens_by_attention_mask,
    get_audio_span_indices,
)


def test_build_audio_code_id_map():
    vocab = {
        "<audio>": 10,
        "</audio>": 11,
        "<1_0>": 101,
        "<2_0>": 102,
        "<3_1>": 103,
        "hello": 5,
    }
    mapping = build_audio_code_id_map(vocab)
    assert mapping == {
        101: (1, 0),
        102: (2, 0),
        103: (3, 1),
    }


def test_filter_tokens_by_attention_mask():
    token_ids = [9, 9, 1, 2, 3]
    attention_mask = [0, 0, 1, 1, 1]
    assert filter_tokens_by_attention_mask(token_ids, attention_mask) == [1, 2, 3]


def test_get_audio_span_indices_with_end_tag():
    audio_start_id = 10
    audio_end_id = 11
    token_ids = [1, audio_start_id, 101, 102, audio_end_id, 7]
    start, end = get_audio_span_indices(token_ids, audio_start_id, audio_end_id)
    assert (start, end) == (2, 4)


def test_get_audio_span_indices_without_end_tag():
    audio_start_id = 10
    audio_end_id = 11
    token_ids = [1, audio_start_id, 101, 102, 103]
    start, end = get_audio_span_indices(token_ids, audio_start_id, audio_end_id)
    assert (start, end) == (2, 5)


def test_extract_audio_codes_q1():
    vocab = {
        "<audio>": 10,
        "</audio>": 11,
        "<5_0>": 105,
        "<7_0>": 107,
    }
    mapping = build_audio_code_id_map(vocab)
    token_ids = [1, 10, 105, 107, 11]
    codes = extract_audio_codes_bqt_from_token_ids(
        token_ids=token_ids,
        num_quantizers=1,
        audio_code_id_map=mapping,
        audio_start_id=10,
        audio_end_id=11,
    )
    assert codes is not None
    assert codes.shape == (1, 1, 2)
    assert torch.equal(codes.cpu(), torch.tensor([[[5, 7]]]))


def test_extract_audio_codes_q2_malformed_chunk_reset():
    vocab = {
        "<audio>": 10,
        "</audio>": 11,
        "<5_0>": 205,
        "<8_1>": 208,
        "<9_0>": 209,
        "<4_1>": 204,
        "noise": 99,
    }
    mapping = build_audio_code_id_map(vocab)
    # valid frame (5,8), malformed break with noise, valid frame (9,4)
    token_ids = [10, 205, 208, 99, 209, 204, 11]
    codes = extract_audio_codes_bqt_from_token_ids(
        token_ids=token_ids,
        num_quantizers=2,
        audio_code_id_map=mapping,
        audio_start_id=10,
        audio_end_id=11,
    )
    assert codes is not None
    # (B, Q, T) => [[ [5,9], [8,4] ]]
    assert torch.equal(codes.cpu(), torch.tensor([[[5, 9], [8, 4]]]))


def test_build_audio_code_id_map_includes_spark_semantic():
    vocab = {
        "<|bicodec_semantic_12|>": 201,
        "<|bicodec_semantic_99|>": 202,
        "<|bicodec_global_5|>": 301,
    }
    mapping = build_audio_code_id_map(vocab)
    assert mapping == {201: (12, 0), 202: (99, 0)}


def test_extract_spark_global_token_ids_prefers_span_between_markers():
    vocab = {
        "<|start_global_token|>": 10,
        "<|end_global_token|>": 11,
        "<|bicodec_global_7|>": 101,
        "<|bicodec_global_9|>": 102,
        "<|bicodec_global_2|>": 103,
    }
    gmap = build_spark_global_id_map(vocab)
    token_ids = [103, 10, 101, 102, 11, 103]
    out = extract_spark_global_token_ids(
        token_ids=token_ids,
        spark_global_id_map=gmap,
        start_global_id=10,
        end_global_id=11,
    )
    assert out is not None
    assert torch.equal(out.cpu(), torch.tensor([[7, 9]]))


def test_extract_spark_global_token_ids_fallback_scan():
    vocab = {
        "<|bicodec_global_4|>": 201,
        "<|bicodec_global_8|>": 202,
    }
    gmap = build_spark_global_id_map(vocab)
    token_ids = [1, 201, 2, 202, 3]
    out = extract_spark_global_token_ids(
        token_ids=token_ids,
        spark_global_id_map=gmap,
        start_global_id=None,
        end_global_id=None,
    )
    assert out is not None
    assert torch.equal(out.cpu(), torch.tensor([[4, 8]]))
