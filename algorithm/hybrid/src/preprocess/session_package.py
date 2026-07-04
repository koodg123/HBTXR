from __future__ import annotations

from typing import Dict, Sequence


def build_session_package(
    *,
    session_key: str,
    user_id: int,
    eye: str,
    session_code: str,
    data_mode: str,
    canonical_name: str,
    frame_source: str,
    protocol_session_index: int | None,
    sensor_size_wh: Sequence[int],
    eye_region_xywh: Sequence[int],
    frame_source_size_wh: Sequence[int],
    event_source_size_wh: Sequence[int],
    events_npz: str,
    frame_index_path: str,
    annotation_store_path: str,
    session_motion_regime_prior: Dict,
    n_frames: int,
    n_labelled_frames: int,
    interpolation_ref: str | None = None,
    interpolation_summary: Dict | None = None,
) -> Dict:
    payload = {
        "session_key": session_key,
        "user_id": int(user_id),
        "subject_id": int(user_id),
        "eye": str(eye),
        "session_code": str(session_code),
        "data_mode": str(data_mode),
        "canonical_name": str(canonical_name),
        "frame_source": str(frame_source),
        "protocol_session_index": protocol_session_index,
        "sensor_size_wh": [int(v) for v in sensor_size_wh],
        "eye_region_xywh": [int(v) for v in eye_region_xywh],
        "frame_source_size_wh": [int(v) for v in frame_source_size_wh],
        "event_source_size_wh": [int(v) for v in event_source_size_wh],
        "events_npz": str(events_npz),
        "frame_index_path": str(frame_index_path),
        "annotation_store_path": str(annotation_store_path),
        "split_policy": "subject_independent_exgaze_with_val",
        "session_motion_regime_prior": dict(session_motion_regime_prior),
        "n_frames": int(n_frames),
        "n_labelled_frames": int(n_labelled_frames),
    }
    if interpolation_ref is not None:
        payload["interpolation_ref"] = str(interpolation_ref)
    if interpolation_summary is not None:
        payload["interpolation_summary"] = dict(interpolation_summary)
    return payload
