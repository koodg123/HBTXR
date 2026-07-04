# Hybrid Provenance

## Sources

This directory combines three source groups:

- legacy hybrid configs moved from the former consolidated algorithm workspace
- `external_hybrid_package`
- `hgtxr_software`

## Imported Files

```text
former algorithm/configs/*hybrid* -> hybrid/configs/legacy/
external_hybrid_package/configs/ -> hybrid/configs/external/
external_hybrid_package/scripts/ -> hybrid/scripts/external_pipeline/
external_hybrid_package/src/hbtxr/ -> hybrid/src/
external_hybrid_package/tests/ -> hybrid/tests/external_pipeline/
hgtxr_software/*.py -> hybrid/hardware_reference/src/
hgtxr_software/modules/ -> hybrid/hardware_reference/src/modules/
hgtxr_software/configs/ -> hybrid/hardware_reference/configs/
hgtxr_software/docs/ -> hybrid/hardware_reference/docs/
hgtxr_software/scripts/ -> hybrid/hardware_reference/scripts/
hgtxr_software/tools/ -> hybrid/hardware_reference/tools/
hgtxr_software/tests/ -> hybrid/hardware_reference/tests/
```

## Role

This directory is for frame-event fusion and runtime policy surfaces:

- search + event + track models
- interpolation and event-generation pipeline code
- runtime search/track scheduler logic
- hardware-aware software reference logic
