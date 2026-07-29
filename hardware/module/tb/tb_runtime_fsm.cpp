#include "../include/common.h"
#include <cstdio>

int main() {
  HGTXRDecision d = runtime_fsm(1, 1, 1, 1, 1, false);
  std::printf("state=%d reason=%d\n", d.state, d.reason);
  return d.state == 1 ? 0 : 1;
}

