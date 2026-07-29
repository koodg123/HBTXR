# Req9 DeiT Image Reference Audit

- status: `pass`
- root: `/home/kjm26/project/PRJXR/XR-VIT/HGTXR`
- requested_user_path: `/home/kjm26/project/PRJXR/XR-VIT\PAPER_PRJXR\05_RESOURCES\DeiT-Tiny C-Syn Results.png`
- normalized_requested_path: `/home/kjm26/project/PRJXR/XR-VIT/PAPER_PRJXR/05_RESOURCES/DeiT-Tiny C-Syn Results.png`
- checks: `11/11`
- fail_count: `0`

## Images

| Role | Exists | Size | SHA256 | Path |
|---|---:|---:|---|---|
| requested_image | `True` | `69712` | `90495791e6103be30268320d3b17dce2eaa52d003dd6f341393262745c1c3b79` | /home/kjm26/project/PRJXR/XR-VIT/PAPER_PRJXR/05_RESOURCES/DeiT-Tiny C-Syn Results.png |
| hgpipe_substitute | `True` | `69712` | `90495791e6103be30268320d3b17dce2eaa52d003dd6f341393262745c1c3b79` | /home/kjm26/project/PRJXR/XR-VIT/HGPIPE/DeiT-Tiny C-Syn Results.png |
| hardware_docs_copy | `True` | `69712` | `90495791e6103be30268320d3b17dce2eaa52d003dd6f341393262745c1c3b79` | /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/docs/DeiT-Tiny C-Syn Results.png |

## Checks

| Check | Status | Detail |
|---|---|---|
| requested_path_normalizes_to_paper_prjxr | `pass` | {'requested_user_suffix': 'PAPER_PRJXR\\05_RESOURCES\\DeiT-Tiny C-Syn Results.png', 'normalized': '/home/kjm26/project/PRJXR/XR-VIT/PAPER_PRJXR/05_RESOURCES/DeiT-Tiny C-Syn Results.png'} |
| requested_image_exists | `pass` | /home/kjm26/project/PRJXR/XR-VIT/PAPER_PRJXR/05_RESOURCES/DeiT-Tiny C-Syn Results.png |
| requested_image_nonempty | `pass` | 69712 |
| requested_image_png_magic | `pass` | 89504e470d0a1a0a |
| requested_image_sha256_valid | `pass` | 90495791e6103be30268320d3b17dce2eaa52d003dd6f341393262745c1c3b79 |
| hgpipe_substitute_exists | `pass` | /home/kjm26/project/PRJXR/XR-VIT/HGPIPE/DeiT-Tiny C-Syn Results.png |
| hgpipe_substitute_png_magic | `pass` | 89504e470d0a1a0a |
| hgpipe_substitute_matches_requested_sha256 | `pass` | {'requested': '90495791e6103be30268320d3b17dce2eaa52d003dd6f341393262745c1c3b79', 'hgpipe': '90495791e6103be30268320d3b17dce2eaa52d003dd6f341393262745c1c3b79'} |
| hardware_docs_copy_exists | `pass` | /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/docs/DeiT-Tiny C-Syn Results.png |
| hardware_docs_copy_png_magic | `pass` | 89504e470d0a1a0a |
| hardware_docs_copy_matches_requested_sha256 | `pass` | {'requested': '90495791e6103be30268320d3b17dce2eaa52d003dd6f341393262745c1c3b79', 'docs_copy': '90495791e6103be30268320d3b17dce2eaa52d003dd6f341393262745c1c3b79'} |

## Safety

- executes_commands: `False`
- executes_network: `False`
- writes_canonical_inputs: `False`
- creates_board_result: `False`
- creates_xr_vits_policy: `False`
