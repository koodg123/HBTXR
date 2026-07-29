#pragma once

// Generated from hardware/refs/e2e_axis_vector_spec.json by validate_e2e_axis_vector.py.
// Keep this file synchronized with the spec before running the C++/Vitis gate.
struct HgtxrPatchEntry { int channel; int raw; };
struct HgtxrQkvEntry { int input; int output; int q; int k; int v; };
struct HgtxrMatrixEntry { int row; int col; int raw; };

static const int kGroupChannels = 6;
static const int kExpectedBlocks = 2;
static const int kExpectedActiveTokens = 16;
static const int kExpectedPatchGridH = 1;
static const int kExpectedPatchGridW = 16;
static const int kExpectedFfDim = 768;
static const int kPatchRaw[6] = {1, 2, -1, 3, -2, 4};
static const int kHeadRaw[6] = {7, -8, 4, -4, 6, -6};
static const int kLnBetaRaw = 7;
static const int kQkvDiagRaw = 7;
static const int kWoGroupRaw = 7;
static const int kMlpW1GroupRaw = 7;
static const int kMlpW2DiagRaw = 7;
static const int kExpectedRuntimeState = 2;
static const int kExpectedRaw[6] = {26, -5, 19, -2, 8, -6};

static const int kPatchExtraEntriesCount = 12;
static const HgtxrPatchEntry kPatchExtraEntries[12] = {
  {6, 2},
  {7, -3},
  {8, 5},
  {9, -1},
  {10, 3},
  {11, -4},
  {12, 1},
  {13, -2},
  {14, 4},
  {15, -3},
  {16, 2},
  {17, -1},
};
static const int kQkvExtraEntriesCount = 12;
static const HgtxrQkvEntry kQkvExtraEntries[12] = {
  {0, 6, 3, -2, 2},
  {1, 7, -3, 2, -1},
  {2, 8, 4, 1, 3},
  {3, 9, -2, -3, 2},
  {4, 10, 2, 3, -2},
  {5, 11, -1, 4, 1},
  {6, 12, 2, -1, 3},
  {7, 13, -2, 3, -3},
  {8, 14, 1, 2, 4},
  {9, 15, -4, -1, 2},
  {10, 16, 3, 1, -2},
  {11, 17, -1, 4, 2},
};
static const int kWoExtraEntriesCount = 12;
static const HgtxrMatrixEntry kWoExtraEntries[12] = {
  {6, 1, 3},
  {7, 2, -2},
  {8, 3, 4},
  {9, 4, -3},
  {10, 5, 2},
  {11, 0, -1},
  {12, 6, 2},
  {13, 7, -2},
  {14, 8, 3},
  {15, 9, -3},
  {16, 10, 4},
  {17, 11, -4},
};
static const int kMlpW1ExtraEntriesCount = 18;
static const HgtxrMatrixEntry kMlpW1ExtraEntries[18] = {
  {0, 6, 2},
  {1, 7, -2},
  {2, 8, 3},
  {3, 9, -3},
  {4, 10, 4},
  {5, 11, -1},
  {6, 128, 2},
  {7, 159, -2},
  {8, 191, 3},
  {9, 224, -3},
  {10, 255, 4},
  {11, 143, -1},
  {12, 256, 2},
  {13, 383, -2},
  {14, 511, 3},
  {15, 512, -3},
  {16, 640, 4},
  {17, 767, -1},
};
static const int kMlpW2ExtraEntriesCount = 18;
static const HgtxrMatrixEntry kMlpW2ExtraEntries[18] = {
  {6, 2, -3},
  {7, 3, 2},
  {8, 4, -2},
  {9, 5, 3},
  {10, 0, 1},
  {11, 1, -4},
  {128, 6, 2},
  {159, 7, -3},
  {191, 8, 3},
  {224, 9, -2},
  {255, 10, 1},
  {143, 11, -4},
  {256, 12, 2},
  {383, 13, -3},
  {511, 14, 3},
  {512, 15, -2},
  {640, 16, 1},
  {767, 17, -4},
};
static const int kHeadExtraEntriesCount = 18;
static const HgtxrMatrixEntry kHeadExtraEntries[18] = {
  {0, 1, 2},
  {1, 2, -3},
  {2, 3, 5},
  {3, 4, -2},
  {4, 5, 3},
  {5, 0, -1},
  {0, 6, 1},
  {1, 7, -2},
  {2, 8, 2},
  {3, 9, -1},
  {4, 10, 3},
  {5, 11, -3},
  {0, 12, 2},
  {1, 13, -1},
  {2, 14, 3},
  {3, 15, -2},
  {4, 16, 1},
  {5, 17, -4},
};
