# ADR

## ADR-001: Read ICCAD24-HG-PIPE As Evidence Only

- Decision: do not modify `../ICCAD24-HG-PIPE`.
- Reason: it is the reference source and already has many modified files.
- Consequence: all reconstruction artifacts live in `HG-PIPE-Quantization`.

## ADR-002: Verify Against Golden Refs

- Decision: define completion for inference quantization as bit-exact agreement with all discovered refs.
- Reason: refs encode the original HLS expected behavior.
- Consequence: generated reports provide stronger evidence than qualitative analysis alone.
