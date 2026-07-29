#pragma once

#include "fixed_types.h"

inline hgtxr_data_t clamp_data(hgtxr_acc_t x) {
#ifndef __SYNTHESIS__
  return (hgtxr_data_t)x;
#else
  if (x > (hgtxr_acc_t)31.0) return (hgtxr_data_t)31.0;
  if (x < (hgtxr_acc_t)-32.0) return (hgtxr_data_t)-32.0;
  return (hgtxr_data_t)x;
#endif
}
