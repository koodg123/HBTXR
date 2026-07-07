# Architecture Decision Records

## ADR-001: Use `software/` as Python package

Decision: Use `software` for the algorithm package name because the user
explicitly requested replacing `hgtxr` with `software`.

## ADR-002: Keep HLS as first hardware source of truth

Decision: Put behavioral hardware under `hardware/hls` and reserve
`hardware/rtl` for generated or wrapper artifacts.

