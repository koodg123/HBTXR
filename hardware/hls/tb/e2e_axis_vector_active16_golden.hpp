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
static const int kExpectedFfDim = 32;
static const int kPatchRaw[6] = {1, 2, -1, 3, -2, 4};
static const int kHeadRaw[6] = {7, -8, 4, -4, 6, -6};
static const int kLnBetaRaw = 7;
static const int kQkvDiagRaw = 7;
static const int kWoGroupRaw = 7;
static const int kMlpW1GroupRaw = 7;
static const int kMlpW2DiagRaw = 7;
static const int kExpectedRuntimeState = 2;
static const int kExpectedRaw[6] = {24, -9, 8, -5, 2, -14};

static const int kPatchExtraEntriesCount = 6;
static const HgtxrPatchEntry kPatchExtraEntries[6] = {
  {6, 2},
  {7, -3},
  {8, 5},
  {9, -1},
  {10, 3},
  {11, -4},
};
static const int kQkvExtraEntriesCount = 6;
static const HgtxrQkvEntry kQkvExtraEntries[6] = {
  {0, 6, 3, -2, 2},
  {1, 7, -3, 2, -1},
  {2, 8, 4, 1, 3},
  {3, 9, -2, -3, 2},
  {4, 10, 2, 3, -2},
  {5, 11, -1, 4, 1},
};
static const int kWoExtraEntriesCount = 6;
static const HgtxrMatrixEntry kWoExtraEntries[6] = {
  {6, 1, 3},
  {7, 2, -2},
  {8, 3, 4},
  {9, 4, -3},
  {10, 5, 2},
  {11, 0, -1},
};
static const int kMlpW1ExtraEntriesCount = 6;
static const HgtxrMatrixEntry kMlpW1ExtraEntries[6] = {
  {0, 6, 2},
  {1, 7, -2},
  {2, 8, 3},
  {3, 9, -3},
  {4, 10, 4},
  {5, 11, -1},
};
static const int kMlpW2ExtraEntriesCount = 6;
static const HgtxrMatrixEntry kMlpW2ExtraEntries[6] = {
  {6, 2, -3},
  {7, 3, 2},
  {8, 4, -2},
  {9, 5, 3},
  {10, 0, 1},
  {11, 1, -4},
};
static const int kHeadExtraEntriesCount = 6;
static const HgtxrMatrixEntry kHeadExtraEntries[6] = {
  {0, 1, 2},
  {1, 2, -3},
  {2, 3, 5},
  {3, 4, -2},
  {4, 5, 3},
  {5, 0, -1},
};
