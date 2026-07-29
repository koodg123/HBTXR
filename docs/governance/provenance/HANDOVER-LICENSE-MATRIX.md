# HANDOVER License Matrix

Local possession and Git history are not permission to redistribute or promote
code, models, data, or generated hardware products. The machine-readable source
of truth is `handover-source-registry.json`.

| Candidate | Registry class | License evidence | Promotion |
|---|---|---|---|
| ERVT, TENNs-Eye, TDTracker, BRAT | `LICENSE_BLOCKED` | No governing mapping verified | Blocked |
| FECET | `REFERENCE_ONLY` | Archive/report provenance only | Blocked |
| Retina | `LICENSE_BLOCKED` | Local LICENSE exists; file/model/data scope unreviewed | Blocked |
| HG-PIPE Quantization | `REFERENCE_ONLY` | No top-level license found | Blocked |
| ICCAD24 HG-PIPE | `REFERENCE_ONLY` | LICENSE exists; reuse/notice scope unreviewed | Blocked |
| HGTXR | `ADAPT` | Local source snapshot; license/rights unverified | Blocked |
| XR_Accel, ViT_Accel | `ADAPT` | Local snapshots; no top-level license found | Blocked |

`ADAPT` describes technical intent, not legal clearance. Promotion requires an
identified upstream, governing license path and scope, required notices, and
separate code/model/data-rights review. Unknown fields remain fail-closed.
