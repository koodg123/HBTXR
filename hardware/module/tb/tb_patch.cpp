// V1 — the patch-embedding stem against patch_y. Conv-F or Conv-E, chosen by the golden.
#include <cstdio>
#include <string>
#include <vector>

#include "hbtxr_config.hpp"
#include "hbtxr_golden.hpp"
#include "hbtxr_patch_embed.hpp"

using namespace hbtxr;
typedef HbtxrCfgPatch PCFG;
typedef HbtxrCfgBase CFG;

static constexpr int D = CFG::D, K = 16, TP = CFG::TP, P = CFG::NL_P;

/// One instantiation per modality, as the paper has them: Conv-F takes a grayscale frame,
/// Conv-E a two-channel event voxel. They differ only in CIN and the input size.
typedef HbtxrPatchEmbed<PCFG, CFG, 1, K, 128, 128, P> conv_f_t;
typedef HbtxrPatchEmbed<PCFG, CFG, 2, K, 64, 64, P> conv_e_t;
static conv_f_t conv_f;
static conv_e_t conv_e;

template <class STEM>
static int run_stem(STEM &stem, const std::string &dir, const char *name, int cin, int hin,
                    int win) {
  const std::vector<long long> img = read_golden(dir, "patch_image");
  const std::vector<long long> w = read_golden(dir, "patch_weight");
  const std::vector<long long> bias = read_golden(dir, "patch_bias_acc");
  const std::vector<long long> zp = read_golden(dir, "patch_zp_correction");
  const std::vector<long long> want = read_golden(dir, "patch_y");
  std::vector<long long> mult, shift;
  read_requant<CFG>(dir, "patch", mult, shift);

  constexpr int TAPS = STEM::TAPS;
  static PCFG::w_t wt[D][TAPS];
  static PCFG::acc_t bt[D];
  static ap_uint<CFG::REQ_M_BITS> mt[D];
  static ap_uint<5> st[D];
  for (int o = 0; o < D; ++o) {
    for (int i = 0; i < TAPS; ++i) wt[o][i] = w[(long long)o * TAPS + i];
    // The zero-point correction is constant per out-channel, so it folds into the bias
    // the accumulator already carries. One resident number, no runtime subtract.
    bt[o] = bias[o] - zp[o];
    if ((long long)bt[o] != bias[o] - zp[o]) {
      std::printf("  %s: bias-zp[%d]=%lld does not fit acc_t (%d bits)\n", name, o,
                  bias[o] - zp[o], (int)PCFG::acc_t::width);
      return 1;
    }
    mt[o] = mult[o];
    st[o] = (int)shift[o];
  }
  stem.pe.load(wt, bt, mt, st);

  // Feed pixel-interleaved [y][x][c]; the golden image is [c][y][x]. On the board this
  // transpose is the DMA descriptor, not logic.
  hls::stream<typename STEM::pix_beat_t> in("img");
  hls::stream<typename STEM::tok_beat_t> out("tok");
  for (int y = 0; y < hin; ++y)
    for (int x0 = 0; x0 < win * cin; x0 += STEM::LANES) {
      typename STEM::pix_beat_t v;
      for (int l = 0; l < STEM::LANES; ++l) {
        const int idx = x0 + l, x = idx / cin, c = idx % cin;
        v[l] = img[((long long)c * hin + y) * win + x];
      }
      in.write(v);
    }

  stem.run(in, out);

  const int tokens = STEM::TOKENS;
  std::vector<long long> got((size_t)tokens * D);
  for (int t0 = 0; t0 < tokens; t0 += TP)
    for (int o0 = 0; o0 < D; o0 += P) {
      const typename STEM::tok_beat_t v = out.read();
      for (int p = 0; p < TP; ++p)
        for (int c = 0; c < P; ++c) got[(t0 + p) * D + o0 + c] = (long long)v[p * P + c];
    }

  std::printf("tb_patch    %s  [%d,%d,%d] -> %d tokens x %d, taps=%d, acc%db\n", name, cin,
              hin, win, tokens, D, TAPS, (int)PCFG::acc_t::width);
  int bad = compare("patch_y", got, want);
  if (!in.empty() || !out.empty()) {
    std::printf("  drain      FAIL  in=%zu out=%zu\n", in.size(), out.size());
    bad += 1;
  } else {
    std::printf("  drain      ok    both streams empty\n");
  }
  bad += expect_reject("negctl", got, want);
  return bad;
}

int main(int argc, char **argv) {
  const std::string dir = argc > 1 ? argv[1] : "hardware/workspace/golden/search-a4";
  const std::vector<long long> meta = read_golden(dir, "meta");
  const int patch = (int)meta[6], cin = (int)meta[7], hin = (int)meta[8], win = (int)meta[9];
  if (patch != K) {
    std::printf("  meta       FAIL  golden patch=%d, unit K=%d\n", patch, K);
    return 1;
  }

  int bad;
  if (cin == 1 && hin == 128 && win == 128) {
    bad = run_stem(conv_f, dir, "Conv-F", cin, hin, win);
  } else if (cin == 2 && hin == 64 && win == 64) {
    bad = run_stem(conv_e, dir, "Conv-E", cin, hin, win);
  } else {
    std::printf("  meta       FAIL  no stem for [%d,%d,%d]\n", cin, hin, win);
    return 1;
  }
  std::printf("%s\n", bad ? "FAIL" : "PASS");
  return bad;
}
