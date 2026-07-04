# Dual-Panel Instruction Annotation Tool

This repository contains a lightweight web-based annotation tool used in our paper **SAM3-I: Segment Anything with Instructions**. The tool is described in **Appendix B (Data Construction Details)** and was developed to support high-quality instruction-grounding annotation, especially for **subset-level / one-to-many** scenarios.

**Main file:** `dual_panel_instruction_anno_tool-EN.html`

## Overview

The interface shows the same sample in two synchronized panels:

- **Left panel (Annotate):** interactive annotation view for placing point prompts on the image
- **Right panel (Reference):** non-interactive reference view for visually verifying the original image without annotation overlays

This design helps annotators select target instances or subsets more accurately while still checking fine visual details from the clean reference image.

## What is annotated

For each image, the tool collects three complementary annotation signals:

1. **Point prompts**  
   Annotators click on the left panel to indicate the intended target instance(s) or subset.

2. **Referring description (with target term)**  
   A simple instruction that explicitly contains the target term.

3. **Reasoning description (without target term)**  
   A semantically equivalent instruction that does **not** explicitly name the target, so the target must be identified through contextual or functional reasoning.

Together, these fields support the unified collection of **simple** and **complex** instruction-grounding annotations.

## Typical workflow

The tool supports a two-stage human workflow described in the paper:

- **Annotation:** annotators place points, write the referring description, and write the reasoning description.
- **Inspection:** reviewers verify whether the selected points and the written descriptions are consistent with the image.

The right reference panel is especially useful during inspection because it shows the original image without overlay interference.

## How to use

### 1. Open the tool

Open `dual_panel_instruction_anno_tool-EN.html` in **Google Chrome**.

> Chrome is recommended because the tool relies on browser support for folder selection.

### 2. Load the two image folders

At the top of the page, load:

- **Left annotation folder:** `A_to_be_annotated`
- **Right reference folder:** `B_for_reference`

Click **Choose Files** for each folder.

### 3. Annotate the target

For each sample:

1. Click inside the target object or subset on the **left panel** to place point prompt(s).
2. Click an existing point again if you want to remove it.
3. Fill in:
   - **Referring Description (with target term)**
   - **Reasoning Description (without target term)**

### 4. Review and navigate

Use **Previous** and **Next** to browse samples.

The tool also provides:

- **Full Screen (Left)** for precise annotation
- **1x / 2x / 3x zoom** in full-screen mode
- **Shift + drag** for panning in full-screen mode
- **Jump by filename** for targeted verification
- **Auto-save** in browser local storage
- **Load Existing JSON** to resume or inspect previous annotations

### 5. Export annotations

After finishing annotation, click:

- **Save annotation.json** to export all samples

You can also click:

- **Export Current JSON** to export only the current sample

## Output

The main output file is:

- `annotation.json`

The exported annotations include point coordinates and paired text descriptions for each sample.

## Notes

- Only the **left panel** accepts annotation clicks.
- The **right panel** is for reference and inspection only.
- Point coordinates are exported in the **original image resolution**, so they remain consistent regardless of display scaling.
- This tool is particularly useful for cases where an instruction refers to a **subset of visually similar instances**.
