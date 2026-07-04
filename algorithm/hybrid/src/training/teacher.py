from __future__ import annotations

import torch
from torch import nn


def ema_update(teacher: nn.Module, student: nn.Module, decay: float) -> None:
    with torch.no_grad():
        teacher_state = teacher.state_dict()
        student_state = student.state_dict()
        for key, teacher_value in teacher_state.items():
            student_value = student_state[key].detach()
            if tuple(teacher_value.shape) != tuple(student_value.shape):
                if teacher_value.ndim != student_value.ndim:
                    continue
                if any(teacher_dim < student_dim for teacher_dim, student_dim in zip(teacher_value.shape, student_value.shape)):
                    continue
                slices = tuple(slice(0, int(dim)) for dim in student_value.shape)
                if torch.is_floating_point(teacher_value):
                    teacher_value[slices].mul_(decay).add_(student_value, alpha=1.0 - decay)
                else:
                    teacher_value[slices].copy_(student_value)
                continue
            if torch.is_floating_point(teacher_value):
                teacher_value.mul_(decay).add_(student_value, alpha=1.0 - decay)
            else:
                teacher_value.copy_(student_value)


__all__ = ["ema_update"]
