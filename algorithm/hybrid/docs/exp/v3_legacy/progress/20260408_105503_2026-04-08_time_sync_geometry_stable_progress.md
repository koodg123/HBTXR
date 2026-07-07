# Time-Synchronous Geometry-Stable Progress

- Date: `2026-04-08`
- Scope: event-view rendering, accumulated event summaries, figure-ready result panels, and frame-to-frame interpolation previews for `user01 / session_201 / interval_5012`

## Summary

This progress batch focused on turning the interval-5012 analysis outputs into figure-ready assets for the "Time-synchronous & Geometry-stable supervision" story.

The work now covers:

- sparse event-only `axis_swapped_views` for interval 5012
- sampled-interval event 3D renders across the selected interval set
- sampled-interval accumulated event summaries
- figure-ready event 3D / accumulation / frame panels
- raw frame-to-frame source strips
- dense frame-to-frame interpolation previews and per-frame exports

## Completed Work

### 1. Event 3D views

- Added `scripts/render_axis_swapped_views.py`
- Regenerated interval-5012 `axis_swapped_views` with:
  - event-only point cloud
  - axes removed
  - ticks removed
  - title/legend/ROI overlays removed
- Reduced the interval-5012 point cloud density for a sparser figure-friendly rendering

Key outputs:

- `workspace_event_voxel_analysis/axis_swapped_views/`
- `workspace_event_voxel_analysis/user01_session201_interval_events_axis_swapped_views/`

### 2. Sampled accumulated event results

- Added `scripts/render_sampled_interval_accumulated_channels.py`
- Generated accumulated negative/positive event images for the sampled interval set
- Built per-group page sheets and summary files

Key outputs:

- `workspace_event_voxel_analysis/user01_session201_interval_events_accumulated_channels/`

### 3. Conceptual and before/after figures

- Added `scripts/generate_time_sync_geometry_stable_figures.py`
- Added `scripts/generate_event_interpolation_before_after_figure.py`
- Exported conceptual and narrative figure bundles in `PNG`, `SVG`, and `PDF`

Key outputs:

- `workspace_figures/time_sync_geometry_stable_figure_set/`
- `workspace_figures/event_interpolation_before_after/`

### 4. Figure-ready result panels

- Added and iteratively updated `scripts/generate_time_sync_geometry_stable_result_panels.py`
- Generated compact figure-ready panels for:
  - frame anchor result crops
  - event 3D result crops
  - event 3D + accumulation pair panels
  - frame-to-frame source strips
  - frame interpolation before/after panels

Key outputs:

- `workspace_figures/time_sync_geometry_stable_result_panels/`

Representative files:

- `frame_interpolation_result_left.png`
- `event_interpolation_result_3d_left.png`
- `event_interpolation_result_3d_plus_accumulate_left.png`
- `frame_to_frame_source_left.png`
- `frame_to_frame_dense_left.png`
- `frame_interpolation_before_after_left.png`
- `frame_interpolation_before_after_pair.png`

### 5. Dense frame export folders

The dense frame sequence is no longer only summarized as a strip preview.

Per-frame exports are now saved into folders:

- `workspace_figures/time_sync_geometry_stable_result_panels/frame_to_frame_dense_left_frames/`
- `workspace_figures/time_sync_geometry_stable_result_panels/frame_to_frame_dense_right_frames/`

Current naming scheme:

- `000_endpoint.png`
- `001_alpha_0p143.png`
- `...`
- `007_endpoint.png`

## Data Sources Used

### Figure crops and event panels

- `workspace_event_voxel_analysis/interval_5012_frame_eye_pupil_mask_overlay_blue/`
- `workspace_event_voxel_analysis/interval_5012_accumulated_channels/`
- `workspace_event_voxel_analysis/interval_5012_accumulated_channels_with_rois/`
- `workspace_event_voxel_analysis/user01_session201_interval_events_axis_swapped_views/`
- `workspace_event_voxel_analysis/user01_session201_interval_events_accumulated_channels/`

### Raw frame pair for frame-to-frame preview

- `/home/jaemyung/project/dataset/EYE/EV_Eye/raw_dataset/Data_davis/user1/left/session_2_0_1/frames/005012_1657711896098367.png`
- `/home/jaemyung/project/dataset/EYE/EV_Eye/raw_dataset/Data_davis/user1/left/session_2_0_1/frames/005013_1657711896138367.png`
- `/home/jaemyung/project/dataset/EYE/EV_Eye/raw_dataset/Data_davis/user1/right/session_2_0_1/frames/005012_1657711896077669.png`
- `/home/jaemyung/project/dataset/EYE/EV_Eye/raw_dataset/Data_davis/user1/right/session_2_0_1/frames/005013_1657711896117669.png`

## Assumptions and Limitations

- The "sampled intervals" were interpreted as the previously selected `uniform / burst / sparse` interval sets.
- The raw frame-to-frame preview uses a fixed eye-region crop for geometric stability across the strip.
- The dense frame interpolation preview currently uses linear blending.
- A TimeLens checkpoint was not found in the current workspace, so model-based frame interpolation was not run in this batch.
- Previously generated `frame_to_frame_dense_left.gif` and `frame_to_frame_dense_right.gif` remain in the output folder as legacy files from an earlier iteration and were not deleted.

## Suggested Next Steps

- Replace linear-blend dense frame previews with a TimeLens-based interpolation pass when a checkpoint is available.
- If the manuscript figure needs stricter styling, regenerate the frame strips with:
  - exact mockup tilt
  - transparent background
  - label-free export
- Decide whether generated workspace outputs should remain local-only or be versioned in git long term.
