import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import REPORTS_DIR
s = pd.read_csv(os.path.join(REPORTS_DIR,"eveye_real_permotion_session_level.csv"))
fig, ax = plt.subplots(1,2, figsize=(11,4.3))
for i,(stat,lab) in enumerate([("median","median px error"),("p95","p95 px error")]):
    piv = s.pivot(index="motion", columns="track", values=stat).reindex(["saccade","smooth"])
    piv.plot(kind="bar", ax=ax[i], color={"frame":"#1f77b4","event":"#d1495b"})
    ax[i].set_title(f"EV-Eye REAL {lab}\n(48 users, session-level)"); ax[i].set_ylabel(lab)
    ax[i].set_xlabel("motion (session type)"); plt.setp(ax[i].get_xticklabels(), rotation=0)
plt.tight_layout(); p=os.path.join(REPORTS_DIR,"fig_eveye_real_permotion.png")
plt.savefig(p, dpi=130); print("wrote", p)
