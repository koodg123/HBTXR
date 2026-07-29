# Resource Extraction Log

## 2026-06-06 Continuation

- Source: `../impl_repos`
- Command: `python3 hardware/tools/extract_resource_metrics.py --impl-root ../impl_repos --out docs/resources/deit_tiny_csyn_resource_timing_metrics.csv --run-id continuation_2026_06_06 --target ZCU104 --fit-goal zcu104_cyclic_transformer`
- Result: 41,302 candidate metric rows plus CSV header.
- Method: regex-first-pass report parsing, `source_confidence=0.7`; values require csynth/XML-focused filtering before final ZCU104 fit claims.
- DeiT image status: `../PAPER_PRJXR/05_RESOURCES/DeiT-Tiny C-Syn Results.png` exists and is a 955x429 PNG, but `tesseract` is not installed and the app image viewer could not open the WSL path. Image extraction remains unverified.
