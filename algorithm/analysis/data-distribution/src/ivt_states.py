"""I-VT 3-state labeling for GT frames.
state = Saccade  if speed > v_sacc
        Fixation if speed<=v_sacc and session is saccade-type (101/201)
        Smooth   if speed<=v_sacc and session is smooth-type (102/202)
v_sacc = 90th percentile of saccade-session speed (data-driven)."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from common import CACHE_DIR

def labeled_velocity(p_sacc=90):
    v = pd.read_csv(os.path.join(CACHE_DIR, "velocity_at_gt.csv"))
    v = v[v.speed_pxps.notna()].copy()
    v_sacc = np.percentile(v[v.motion=="saccade"].speed_pxps, p_sacc)
    v["state"] = np.where(v.speed_pxps > v_sacc, "Saccade",
                   np.where(v.motion=="saccade", "Fixation", "Smooth"))
    return v, float(v_sacc)

if __name__ == "__main__":
    v, vs = labeled_velocity()
    print("v_sacc=%.0f px/s"%vs)
    print(v.state.value_counts().to_string())
