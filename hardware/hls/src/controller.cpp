#include "../include/common.h"

int controller_active_tokens(int mode) {
  return mode == HGTXR_MODE_TRACK ? HGTXR_TRACK_TOKENS : HGTXR_SEARCH_TOKENS;
}

int controller_depth_limit(int mode) {
  return mode == HGTXR_MODE_TRACK ? HGTXR_TRACK_CUT_DEPTH : HGTXR_SEARCH_DEPTH;
}
