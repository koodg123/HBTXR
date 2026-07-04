from software.config import load_config
from software.dataset import make_synthetic_batch
from software.model import build_model
from software.runtime import RuntimeHGTXRTracker
from software.scheduler import TrackSearchSchedulerFSM


def test_scheduler_track_decision():
    fsm = TrackSearchSchedulerFSM()
    d = fsm.step(search_conf=1, track_conf=1, track_quality=1, similarity=1, event_density=1, closed_eye_flag=False)
    assert d.state == "track"


def test_runtime_step():
    cfg = load_config("software/configs/base.yaml")
    cfg["model"]["embed_dim"] = 24
    cfg["model"]["depth"] = 1
    cfg["model"]["num_heads"] = 3
    model = build_model(cfg)
    tracker = RuntimeHGTXRTracker(model)
    batch = make_synthetic_batch(1)
    out = tracker.step(frame=batch["frame"], event=batch["event"], event_density=batch["event_density"])
    assert "runtime/state" in out

