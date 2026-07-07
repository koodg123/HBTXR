# Resource Extraction Protocol

Date: 2026-06-06
Scope: `PAPER_PRJXR/05_RESOURCES/DeiT-Tiny C-Syn Results.png` and `impl_repos`

## Purpose

Extract resource, latency, and timing targets from the DeiT-Tiny C-synthesis reference image and prior implementation repositories, then normalize them into a versioned metric table for ZCU104-fit decisions.

## Checklist

1. Create a run registry entry:
   - `run_id`
   - date
   - branch/commit
   - board target
   - fit goal
   - extractor version
   - operator
2. Define metric scope before extraction:
   - resource
   - latency
   - timing
3. Inspect the image first:
   - save ROI/crop path per visible table or plot
   - record OCR confidence if OCR is used
   - mark manual corrections as `image_manual`
4. Parse `impl_repos` reports deterministically:
   - `.rpt`
   - `.xml`
   - `.syr`
   - `.log`
   - generated CSV/JSON reports if present
5. Normalize units immediately:
   - LUT/FF/DSP as absolute counts
   - BRAM as BRAM_18K blocks unless the source says otherwise
   - URAM as blocks
   - timing in ns and MHz
   - latency in cycles and ms when clock is known
6. Capture provenance for every metric:
   - `image_ocr`
   - `image_manual`
   - `report_parsed`
   - `derived`
7. Reconcile conflicts:
   - if two sources differ by more than 1%, mark `needs_review`
   - if units are ambiguous, mark `needs_review`
   - prefer parser-verified values over image values unless the image contains a unique metric
8. Store rejected or ambiguous values in an exception list.
9. Run consistency checks:
   - utilization percent recomputed from used/available
   - `period_ns * fmax_mhz ~= 1000`
   - throughput and latency consistency
10. Freeze the snapshot:
   - CSV
   - checksum
   - extraction log

## Source Type Policy

| Source type | Meaning | Use in final comparison |
|---|---|---|
| `image_ocr` | OCR text from the reference image | provisional until manually checked |
| `image_manual` | human-corrected value from the image | acceptable if no report exists |
| `report_parsed` | deterministic parser output from reports | preferred |
| `derived` | calculated from other captured metrics | acceptable with formula noted |

## Minimum Metrics

Resource:

- `LUT`
- `FF`
- `DSP`
- `BRAM_18K`
- `URAM`
- `Clocking`

Latency:

- `Latency_cycles`
- `Latency_ms`
- `Throughput_fps` or `Throughput_ips`

Timing:

- `Target_Clk_MHz`
- `Achieved_Clk_MHz`
- `Clk_Period_ns`
- `WNS_ns`
- `TNS_ns`

## Output Files

- `docs/resources/deit_tiny_csyn_resource_timing_metrics.csv`
- `docs/resources/deit_tiny_csyn_exceptions.csv`
- `docs/resources/resource_extraction_log.md`

