# HGTXR Reference Input Audit

- status: `needs-reference-input`
- approved_replacements: `False`
- blockers: `1`

## Blockers

- `requested XR-VITs sibling`: requested directory missing
  - path: `/home/kjm26/project/PRJXR/XR-VITs`
  - candidates_found: `3`
  - action: restore requested checkout or get explicit approval for a replacement source

## Image Candidates

- `requested` exists=`True` path=`/home/kjm26/project/PRJXR/XR-VIT/PAPER_PRJXR/05_RESOURCES/DeiT-Tiny C-Syn Results.png`, sha256=`90495791e6103be30268320d3b17dce2eaa52d003dd6f341393262745c1c3b79`
- `candidate-hgpipe` exists=`True` path=`/home/kjm26/project/PRJXR/XR-VIT/HGPIPE/DeiT-Tiny C-Syn Results.png`, sha256=`90495791e6103be30268320d3b17dce2eaa52d003dd6f341393262745c1c3b79`

## HLS Code Candidates

- `requested` exists=`False` path=`/home/kjm26/project/PRJXR/XR-VITs` markers=`none`
- `candidate-xr-accel` exists=`True` path=`/home/kjm26/project/PRJXR/XR-VIT/XR_Accel` markers=`README.md`
- `candidate-analysis-xr-accel` exists=`True` path=`/home/kjm26/project/PRJXR/XR-VIT/analysis/XR_Accel` markers=`README.md`
- `candidate-vit-accel` exists=`True` path=`/home/kjm26/project/PRJXR/XR-VIT/ViT_Accel` markers=`README.md`

## Restore Commands

```sh
mkdir -p /home/kjm26/project/PRJXR/XR-VIT/PAPER_PRJXR/05_RESOURCES
cp -a /home/kjm26/project/PRJXR/XR-VIT/HGPIPE/DeiT-Tiny\ C-Syn\ Results.png /home/kjm26/project/PRJXR/XR-VIT/PAPER_PRJXR/05_RESOURCES/DeiT-Tiny\ C-Syn\ Results.png
# restore or clone the requested /home/kjm26/project/PRJXR/XR-VITs checkout
```
