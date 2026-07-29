#include "../include/common.h"

void weight_prefetcher(int mode, int block_idx, int *prefetch_valid) {
  // Search weights may be fetched on demand; track-resident weights are treated
  // as always resident for the reduced-depth path.
  *prefetch_valid = (mode == HGTXR_MODE_TRACK || block_idx >= 0) ? 1 : 0;
}
