#pragma once

#include "fixed_types.h"

struct HGTXRState {
  hgtxr_data_t v[6];
};

struct HGTXRDecision {
  int state;
  int reason;
};

