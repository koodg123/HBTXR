#include "../include/common.h"

HGTXRDecision runtime_fsm(hgtxr_data_t search_conf, hgtxr_data_t track_conf, hgtxr_data_t track_quality, hgtxr_data_t similarity, hgtxr_data_t event_density, bool closed_eye) {
  HGTXRDecision d;
  if (closed_eye) {
    d.state = 0;
    d.reason = 1;
    return d;
  }
  if (event_density < (hgtxr_data_t)0.002) {
    d.state = 0;
    d.reason = 2;
    return d;
  }
  if (track_conf >= (hgtxr_data_t)0.45 && track_quality >= (hgtxr_data_t)0.45 && similarity >= (hgtxr_data_t)0.5) {
    d.state = 1;
    d.reason = 3;
    return d;
  }
  d.state = search_conf >= (hgtxr_data_t)0.35 ? 0 : 0;
  d.reason = 4;
  return d;
}

