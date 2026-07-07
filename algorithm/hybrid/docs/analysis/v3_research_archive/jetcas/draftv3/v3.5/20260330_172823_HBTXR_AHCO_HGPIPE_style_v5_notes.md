# HBTXR AHCO/HGPIPE-style revision v5

This revision reflects the requested manuscript updates.

## Main updates

1. **Table IV / Section V-A**
   - The experimental configuration is now described in prose in Section V-A.
   - Table IV is kept as a compact pointer table rather than a dense configuration table.

2. **Target platform**
   - The FPGA target is set to **ZCU104** throughout the hardware and experiment sections.

3. **Resource Utilization table added**
   - Added **Table VI** for ZCU104 resource reporting.
   - The structure includes LUT / FF / BRAM / DSP / clock / board power / mode split.
   - Unsupported post-route values remain `TBD` rather than being fabricated.

4. **Table VII rebuilt**
   - Reorganized as a source-aligned eye-tracking algorithm comparison.
   - Comparison columns are aligned across modality, dataset, output, accuracy, parameters, complexity, and runtime.

5. **Table VIII rebuilt**
   - Reorganized in a deployment-oriented metric-by-system style.
   - Rows include Platform, Dataset, Clock, Resource Utilization, Accuracy, Latency, Power Consumption, Throughput, Power Efficiency, and Frame Efficiency.
   - HBTXR is split into **Search / Track / Scheduled hybrid** columns.

6. **Prose instead of table-only notes**
   - The previous "Positioning w.r.t. HBTXR" style explanation is moved into the paragraph below Table VIII.
   - Scheduled-mode interpretation is also explained in prose.

7. **Section IV-B updates**
   - Section IV-B.1 now explains **dyadic quantization** in an integer-only deployment style.
   - Section IV-B.2 explains each approximation item with text and equations.

8. **Section IV hardware narrative strengthened**
   - The hardware section now explicitly discusses:
     - dataflow
     - memory organization
     - computation units
     - scheduling
     - tiling
     - parallelism
     - multi-stage pipeline
     - cross-layer coupling
     - streaming-oriented feature delivery

9. **Search / Track behavior clarified**
   - Search uses the **full backbone**.
   - Track uses the **front half of the backbone** and then exits early to the Track head.

## Remaining placeholders

- The current source draft does **not** provide finalized post-route HBTXR LUT / FF / BRAM / DSP numbers.
- The current source draft does **not** provide finalized board-level HBTXR power numbers.
- These slots are intentionally kept as `TBD`.
