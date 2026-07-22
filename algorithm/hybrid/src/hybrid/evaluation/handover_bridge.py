"""Explicit coordinate conversion without runtime dependencies."""

from __future__ import annotations

from .contracts import (
    BBox,
    Center,
    Domain,
    Ellipse,
    EvaluationContractError,
    EvaluationTask,
    Size,
    validate_domain_size,
)

EvaluationRecord = Center | BBox | Ellipse
_TASK_TYPES = {
    EvaluationTask.CENTER: Center,
    EvaluationTask.BBOX: BBox,
    EvaluationTask.ELLIPSE: Ellipse,
}


def convert_record(
    record: EvaluationRecord,
    target_domain: Domain,
    target_size: Size,
    task: EvaluationTask,
) -> EvaluationRecord:
    if not isinstance(record, (Center, BBox, Ellipse)):
        raise EvaluationContractError("unsupported evaluation record")
    if not isinstance(task, EvaluationTask):
        raise EvaluationContractError(f"unsupported task: {task!r}")
    validate_domain_size(record.domain, record.size)
    validate_domain_size(target_domain, target_size)
    if not isinstance(record, _TASK_TYPES[task]):
        raise EvaluationContractError(f"task {task.value} and record type mismatch")

    scale_x = target_size.width / record.size.width
    scale_y = target_size.height / record.size.height
    if isinstance(record, Center):
        return Center(target_domain, target_size, record.x * scale_x, record.y * scale_y)
    if isinstance(record, BBox):
        return BBox(
            target_domain, target_size,
            record.x * scale_x, record.y * scale_y,
            record.width * scale_x, record.height * scale_y,
        )
    return Ellipse(
        target_domain, target_size,
        record.cx * scale_x, record.cy * scale_y,
        record.axis_x * scale_x, record.axis_y * scale_y, record.theta,
    )
