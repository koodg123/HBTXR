#include "../include/common.h"
#include <cstdio>

void hgtxr_search_profile_top(
    const hgtxr_data_t frame[HGTXR_HEIGHT][HGTXR_WIDTH],
    hgtxr_data_t out_logits[HGTXR_SEARCH_LOGITS],
    int *runtime_state);

void hgtxr_track_profile_top(
    const hgtxr_data_t event_pos[HGTXR_HEIGHT][HGTXR_WIDTH],
    const hgtxr_data_t event_neg[HGTXR_HEIGHT][HGTXR_WIDTH],
    const hgtxr_data_t frame_pooled[HGTXR_EMBED],
    const hgtxr_data_t prev_state[HGTXR_STATE],
    hgtxr_data_t out_state[HGTXR_STATE],
    int *runtime_state);

int main() {
  static hgtxr_data_t frame[HGTXR_HEIGHT][HGTXR_WIDTH];
  static hgtxr_data_t event_pos[HGTXR_HEIGHT][HGTXR_WIDTH];
  static hgtxr_data_t event_neg[HGTXR_HEIGHT][HGTXR_WIDTH];
  static hgtxr_data_t frame_pooled[HGTXR_EMBED];

  for (int y = 0; y < HGTXR_HEIGHT; ++y) {
    for (int x = 0; x < HGTXR_WIDTH; ++x) {
      frame[y][x] = static_cast<hgtxr_data_t>(((x + y) & 31) * 0.03125f);
      event_pos[y][x] = static_cast<hgtxr_data_t>(((x * 3 + y) & 15) * 0.015625f);
      event_neg[y][x] = static_cast<hgtxr_data_t>(((x + y * 5) & 7) * 0.015625f);
    }
  }
  for (int c = 0; c < HGTXR_EMBED; ++c) {
    frame_pooled[c] = static_cast<hgtxr_data_t>((c & 15) * 0.03125f);
  }

  hgtxr_data_t search_logits[HGTXR_SEARCH_LOGITS];
  hgtxr_data_t prev_state[HGTXR_STATE] = {128, 128, 40, 24, 1, 0};
  hgtxr_data_t track_state[HGTXR_STATE];
  int search_state = -1;
  int track_state_code = -1;

  hgtxr_search_profile_top(frame, search_logits, &search_state);
  hgtxr_track_profile_top(event_pos, event_neg, frame_pooled, prev_state,
                          track_state, &track_state_code);

  if (search_state != HGTXR_MODE_SEARCH) {
    std::printf("search_state mismatch: %d\n", search_state);
    return 1;
  }
  if (track_state_code != HGTXR_MODE_TRACK) {
    std::printf("track_state mismatch: %d\n", track_state_code);
    return 1;
  }
  bool finiteish = true;
  for (int i = 0; i < HGTXR_SEARCH_LOGITS; ++i) {
    finiteish = finiteish && search_logits[i] == search_logits[i];
  }
  for (int i = 0; i < HGTXR_STATE; ++i) {
    finiteish = finiteish && track_state[i] == track_state[i];
  }
  if (!finiteish) {
    std::printf("non-finite output detected\n");
    return 1;
  }

  std::printf("search_state=%d search0=%f search1=%f\n", search_state,
              static_cast<float>(search_logits[0]),
              static_cast<float>(search_logits[1]));
  std::printf("track_state=%d track0=%f track1=%f\n", track_state_code,
              static_cast<float>(track_state[0]),
              static_cast<float>(track_state[1]));
  return 0;
}
