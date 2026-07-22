from __future__ import annotations

from typing import Dict, List, Optional


_PROTOCOL_INDEX_BY_SESSION_DIR = {
    "session_1_0_1": 1,
    "session_1_0_2": 2,
    "session_2_0_1": 3,
    "session_2_0_2": 4,
}

_PROTOCOL_INDEX_BY_SESSION_CODE = {
    "101": 1,
    "102": 2,
    "201": 3,
    "202": 4,
}

_PRIOR_LABEL_BY_PROTOCOL_INDEX = {
    1: "fixation",
    2: "saccade",
    3: "smooth_pursuit",
    4: "smooth_pursuit",
}


def derive_protocol_session_index(session_dir_name: str, session_code: str | None = None) -> tuple[Optional[int], List[str]]:
    if session_dir_name in _PROTOCOL_INDEX_BY_SESSION_DIR:
        return _PROTOCOL_INDEX_BY_SESSION_DIR[session_dir_name], []
    if session_code in _PROTOCOL_INDEX_BY_SESSION_CODE:
        return _PROTOCOL_INDEX_BY_SESSION_CODE[str(session_code)], ["protocol_index_from_session_code"]
    return None, ["protocol_session_index_unknown"]


def derive_session_motion_regime_prior(protocol_session_index: Optional[int]) -> Dict:
    label = _PRIOR_LABEL_BY_PROTOCOL_INDEX.get(protocol_session_index)
    if label is None:
        return {
            "label": "unknown",
            "source": "protocol",
            "confidence": 0.0,
        }
    return {
        "label": label,
        "source": "protocol",
        "confidence": 1.0,
    }
