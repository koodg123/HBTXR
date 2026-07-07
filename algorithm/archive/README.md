# Algorithm Archive

This directory stores imported or legacy algorithm material that is useful for
provenance, comparison, and future porting, but is not part of the active
runtime package.

## Layout

- `imports/legacy_hybrid/`: historical hybrid experiment documents, scripts,
  configs, and reference tests imported from the older hybrid branch.
- `imports/hgtxr/`: HGTXR project documents imported for planning, paper
  mapping, validation, and context.
- `imports/conversations/`: project-classified conversation and progress logs
  reorganized by date.

## Promotion Rule

Files in this archive should not be imported by active code directly. When an
archived script, config, or test becomes useful, port it into the maintained
`algorithm/frame`, `algorithm/event`, or `algorithm/hybrid` surface with a small
adapter commit and a validation note.
