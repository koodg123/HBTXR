from __future__ import annotations

import numpy as np

E2E_WEIGHT_WORDS_256B = 381024
U32_PER_256B_WORD = 8
WEIGHT_BITS = 4
WEIGHT_LANES_PER_256B = 64
EMBED = 192
PATCH = 16
PATCH_ELEMS = PATCH * PATCH
BLOCKS = 6
FF_DIM = 768
STATE = 6

PATCH_WEIGHT_ELEM_BASE = 0
PATCH_WEIGHT_ELEMS = EMBED * PATCH_ELEMS
BLOCK_WEIGHT_ELEM_BASE = PATCH_WEIGHT_ELEM_BASE + PATCH_WEIGHT_ELEMS
LN_PARAM_ELEMS = EMBED
DENSE_MODEL_ELEMS = EMBED * EMBED
MLP_W1_ELEMS = EMBED * FF_DIM
MLP_W2_ELEMS = FF_DIM * EMBED
BLOCK_LN1_GAMMA = 0
BLOCK_LN1_BETA = BLOCK_LN1_GAMMA + LN_PARAM_ELEMS
BLOCK_LN2_GAMMA = BLOCK_LN1_BETA + LN_PARAM_ELEMS
BLOCK_LN2_BETA = BLOCK_LN2_GAMMA + LN_PARAM_ELEMS
BLOCK_WQ = BLOCK_LN2_BETA + LN_PARAM_ELEMS
BLOCK_WK = BLOCK_WQ + DENSE_MODEL_ELEMS
BLOCK_WV = BLOCK_WK + DENSE_MODEL_ELEMS
BLOCK_WO = BLOCK_WV + DENSE_MODEL_ELEMS
BLOCK_W1 = BLOCK_WO + DENSE_MODEL_ELEMS
BLOCK_W2 = BLOCK_W1 + MLP_W1_ELEMS
BLOCK_WEIGHT_ELEMS = BLOCK_W2 + MLP_W2_ELEMS
HEAD_WEIGHT_ELEM_BASE = BLOCK_WEIGHT_ELEM_BASE + BLOCKS * BLOCK_WEIGHT_ELEMS
HEAD_WEIGHT_ELEMS = STATE * EMBED
REQUIRED_WEIGHT_ELEMS = HEAD_WEIGHT_ELEM_BASE + HEAD_WEIGHT_ELEMS
REQUIRED_WEIGHT_WORDS_256B = (REQUIRED_WEIGHT_ELEMS + WEIGHT_LANES_PER_256B - 1) // WEIGHT_LANES_PER_256B

GROUP_CHANNELS = 6
PATCH_RAW = [1, 2, -1, 3, -2, 4]
HEAD_RAW = [7, -8, 4, -4, 6, -6]
LN_BETA_RAW = 7
QKV_DIAG_RAW = 7
WO_GROUP_RAW = 7
MLP_W1_GROUP_RAW = 7
MLP_W2_DIAG_RAW = 7
EXPECTED_RUNTIME_STATE = 2
EXPECTED_RAW = [32, -13, 26, -6, 14, -11]

PATCH_EXTRA_ENTRIES = [
    (6, 2),
    (7, -3),
    (8, 5),
    (9, -1),
    (10, 3),
    (11, -4),
    (12, 1),
    (13, -2),
    (14, 4),
    (15, -3),
    (16, 2),
    (17, -1),
]
QKV_EXTRA_ENTRIES = [
    (0, 6, 3, -2, 2),
    (1, 7, -3, 2, -1),
    (2, 8, 4, 1, 3),
    (3, 9, -2, -3, 2),
    (4, 10, 2, 3, -2),
    (5, 11, -1, 4, 1),
    (6, 12, 2, -1, 3),
    (7, 13, -2, 3, -3),
    (8, 14, 1, 2, 4),
    (9, 15, -4, -1, 2),
    (10, 16, 3, 1, -2),
    (11, 17, -1, 4, 2),
]
WO_EXTRA_ENTRIES = [
    (6, 1, 3),
    (7, 2, -2),
    (8, 3, 4),
    (9, 4, -3),
    (10, 5, 2),
    (11, 0, -1),
    (12, 6, 2),
    (13, 7, -2),
    (14, 8, 3),
    (15, 9, -3),
    (16, 10, 4),
    (17, 11, -4),
]
MLP_W1_EXTRA_ENTRIES = [
    (0, 6, 2),
    (1, 7, -2),
    (2, 8, 3),
    (3, 9, -3),
    (4, 10, 4),
    (5, 11, -1),
    (6, 128, 2),
    (7, 159, -2),
    (8, 191, 3),
    (9, 224, -3),
    (10, 255, 4),
    (11, 143, -1),
    (12, 256, 2),
    (13, 383, -2),
    (14, 511, 3),
    (15, 512, -3),
    (16, 640, 4),
    (17, 767, -1),
]
MLP_W2_EXTRA_ENTRIES = [
    (6, 2, -3),
    (7, 3, 2),
    (8, 4, -2),
    (9, 5, 3),
    (10, 0, 1),
    (11, 1, -4),
    (128, 6, 2),
    (159, 7, -3),
    (191, 8, 3),
    (224, 9, -2),
    (255, 10, 1),
    (143, 11, -4),
    (256, 12, 2),
    (383, 13, -3),
    (511, 14, 3),
    (512, 15, -2),
    (640, 16, 1),
    (767, 17, -4),
]
HEAD_EXTRA_ENTRIES = [
    (0, 1, 2),
    (1, 2, -3),
    (2, 3, 5),
    (3, 4, -2),
    (4, 5, 3),
    (5, 0, -1),
    (0, 6, 1),
    (1, 7, -2),
    (2, 8, 2),
    (3, 9, -1),
    (4, 10, 3),
    (5, 11, -3),
    (0, 12, 2),
    (1, 13, -1),
    (2, 14, 3),
    (3, 15, -2),
    (4, 16, 1),
    (5, 17, -4),
]


def _nibble(value: int) -> int:
    return int(value) & 0xF


def set_q4_weight(weights_u32: np.ndarray, elem_offset: int, value: int) -> None:
    word_idx = elem_offset // WEIGHT_LANES_PER_256B
    lane_idx = elem_offset - word_idx * WEIGHT_LANES_PER_256B
    u32_idx = word_idx * U32_PER_256B_WORD + lane_idx // 8
    shift = (lane_idx % 8) * WEIGHT_BITS
    mask = np.uint32(0xF << shift)
    weights_u32[u32_idx] = np.uint32((int(weights_u32[u32_idx]) & ~int(mask)) | (_nibble(value) << shift))


def get_q4_weight_raw(weights_u32: np.ndarray, elem_offset: int) -> int:
    word_idx = elem_offset // WEIGHT_LANES_PER_256B
    lane_idx = elem_offset - word_idx * WEIGHT_LANES_PER_256B
    u32_idx = word_idx * U32_PER_256B_WORD + lane_idx // 8
    shift = (lane_idx % 8) * WEIGHT_BITS
    raw = (int(weights_u32[u32_idx]) >> shift) & 0xF
    return raw - 16 if raw & 0x8 else raw


def build_active196_b6_ff768_weights() -> np.ndarray:
    weights = np.zeros((E2E_WEIGHT_WORDS_256B * U32_PER_256B_WORD,), dtype=np.uint32)

    for c in range(GROUP_CHANNELS):
        for p in range(PATCH_ELEMS):
            set_q4_weight(weights, PATCH_WEIGHT_ELEM_BASE + c * PATCH_ELEMS + p, PATCH_RAW[c])
        set_q4_weight(weights, HEAD_WEIGHT_ELEM_BASE + c * EMBED + c, HEAD_RAW[c])

    for channel, raw in PATCH_EXTRA_ENTRIES:
        for p in range(PATCH_ELEMS):
            set_q4_weight(weights, PATCH_WEIGHT_ELEM_BASE + channel * PATCH_ELEMS + p, raw)

    for row, col, raw in HEAD_EXTRA_ENTRIES:
        set_q4_weight(weights, HEAD_WEIGHT_ELEM_BASE + row * EMBED + col, raw)

    for block in range(BLOCKS):
        block_base = BLOCK_WEIGHT_ELEM_BASE + block * BLOCK_WEIGHT_ELEMS
        for c in range(EMBED):
            group = c % GROUP_CHANNELS
            set_q4_weight(weights, block_base + BLOCK_LN1_GAMMA + c, 0)
            set_q4_weight(weights, block_base + BLOCK_LN1_BETA + c, LN_BETA_RAW)
            set_q4_weight(weights, block_base + BLOCK_LN2_GAMMA + c, 0)
            set_q4_weight(weights, block_base + BLOCK_LN2_BETA + c, LN_BETA_RAW)
            set_q4_weight(weights, block_base + BLOCK_WQ + c * EMBED + c, QKV_DIAG_RAW)
            set_q4_weight(weights, block_base + BLOCK_WK + c * EMBED + c, QKV_DIAG_RAW)
            set_q4_weight(weights, block_base + BLOCK_WV + c * EMBED + c, QKV_DIAG_RAW)
            set_q4_weight(weights, block_base + BLOCK_WO + c * EMBED + group, WO_GROUP_RAW)
            set_q4_weight(weights, block_base + BLOCK_W1 + c * FF_DIM + group, MLP_W1_GROUP_RAW)

        for h in range(GROUP_CHANNELS):
            set_q4_weight(weights, block_base + BLOCK_W2 + h * EMBED + h, MLP_W2_DIAG_RAW)

        for row, col, q, k, v in QKV_EXTRA_ENTRIES:
            set_q4_weight(weights, block_base + BLOCK_WQ + row * EMBED + col, q)
            set_q4_weight(weights, block_base + BLOCK_WK + row * EMBED + col, k)
            set_q4_weight(weights, block_base + BLOCK_WV + row * EMBED + col, v)

        for row, col, raw in WO_EXTRA_ENTRIES:
            set_q4_weight(weights, block_base + BLOCK_WO + row * EMBED + col, raw)

        for row, col, raw in MLP_W1_EXTRA_ENTRIES:
            set_q4_weight(weights, block_base + BLOCK_W1 + row * FF_DIM + col, raw)

        for row, col, raw in MLP_W2_EXTRA_ENTRIES:
            set_q4_weight(weights, block_base + BLOCK_W2 + row * EMBED + col, raw)

    set_q4_weight(weights, 0, EXPECTED_RUNTIME_STATE - 1)
    return weights
