from hbtxr.models.pruning import resolve_model_role_cfg


def test_teacher_role_accepts_model_teacher_override() -> None:
    cfg = {
        "model": {
            "embed_dim": 256,
            "depth": 8,
            "num_heads": 4,
            "teacher": {
                "embed_dim": 192,
                "depth": 6,
                "num_heads": 3,
                "head_hidden_dim": 192,
            },
        },
        "pruning": {"enabled": False},
    }

    student = resolve_model_role_cfg(cfg, role="student")
    teacher = resolve_model_role_cfg(cfg, role="teacher")

    assert student["embed_dim"] == 256
    assert student["depth"] == 8
    assert teacher["embed_dim"] == 192
    assert teacher["depth"] == 6
    assert teacher["num_heads"] == 3
    assert teacher["head_hidden_dim"] == 192
