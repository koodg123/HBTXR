# Specification

## Scope

The implementation reconstructs the inference-time HG-PIPE quantization flow represented by the public codebase artifacts.

## Operators

- `requant_table`: table-based ReQuant from `src/quant.h`.
- `gelu_requant_table`: fused GeLU/ReQuant from `src/gelu.h`.
- `layernorm_rsqrt_table`: LayerNorm integer mean, variance, rsqrt table, affine, shift, clamp from `src/layernorm.h`.
- `softmax_segmented_table`: Softmax inverse-exp and segmented reciprocal tables from `src/softmax.h`.

## Artifact Contracts

- Scalars and tables are parsed from C include-style `.txt` files in `case/refs`.
- Golden inputs and outputs are parsed from matching `*_input.txt` and `*_output.txt`.
- Type and range metadata are loaded from `statistics/type.npy` and `statistics/range.npy`.

## Non-Scope

- QAT training, calibration dataset generation, and model export are not available in this checkout. This is marked as `확실하지 않음` for any claim beyond inference artifact reconstruction.

## Graph Reconstruction

The graph runner reconstructs patch embedding, 12 attention blocks, 12 MLP blocks, and the head from integer artifacts. It supports component-level verification and single-input end-to-end verification from patch_embed input through head.

## ImageNet Input Bridge

The input bridge converts normalized image tensors into signed int8 patch vectors with the HG-PIPE cls-slot convention. The bridge requires an explicit scale or separately validated calibration policy because the original image-to-patch quantization generation flow is not available in this checkout.

## Quant Parameter Schema

quant_params.py provides TensorDTypeSpec, TensorRangeSpec, AffineQuantParams, LutQuantParams, OpQuantContract, and QuantParamStore. HG-PIPE LUT contracts preserve b, s, bound, and table values directly. Zero-point is represented only for affine quantization contracts; HG-PIPE LUT contracts keep zero_point as None.

## Torch Integer Runtime

int_infer implements a torch integer runner for the artifact graph. The runner uses torch.int64 tensors for deterministic integer accumulation and verifies the same patch_embed input through head chain as the NumPy artifact runner.

## FakeQuantizer Runtime

fake_quant implements AffineFakeQuantizer, HGTableFakeQuantizer, and an FX insertion utility that inserts fake quantizer modules after selected call_module nodes. HGTableFakeQuantizer applies the HG-PIPE LUT cursor formula and returns floating tensors so it can remain inside a fake-quant PyTorch graph.

## Trace Comparison

trace.py defines the shared TensorTrace schema. FakeQuantRunner and TorchIntCaseRunner emit compatible value-bearing traces for LUT-backed cases. compare-traces compares traces by tensor name and writes JSON plus Markdown reports with mismatch count, max absolute error, and mean absolute error.

## FakeQuant Graph Runner

FakeQuantGraphRunner subclasses the torch integer artifact graph runner and routes every HG-PIPE LUT quantization point through HGTableFakeQuantizer. The FakeQuantizer outputs remain observable as floating tensors for trace reporting, then are rounded back to integer tensors so downstream artifact integer operators can continue. This supports verify-fakequant-graph and trace-fakequant-graph.

## Inference Result Comparison

run_result.py defines a compact inference result schema for final logits, including dtype, shape, min/max/mean, top-k entries, and full values. The CLI commands run-int and run-fakequant-graph execute the artifact-backed torch.int and FakeQuantGraph runners from the patch input, then compare-run-results checks exact value equality and top-1 agreement.

## Contract Export API

HgPipeQuantizationPackage is the high-level package entry point. It exposes discovered cases, structured contracts, reference verification, torch.int graph verification, FakeQuant graph verification, final runner comparison, and FakeQuant graph traces. export_contracts and the export-contracts CLI emit JSON-serializable scalar/LUT contracts. LUT contracts include offset, shift_scale, effective_divisor, bound, zero_point, dtype metadata, range metadata, and table sizes by default; --include-tables emits full LUT values. zero_point remains None for recovered HG-PIPE LUT operators because the source artifacts do not contain affine zero-point values for those operators.
