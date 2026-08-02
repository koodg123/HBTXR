// V1 — the whole accelerator, both modes, against the model golden's head accumulator.
//
// Image in, five numbers out. Everything S2..S7 built, routed by one mode bit.
#include <cstdio>
#include <string>
#include <vector>

#include "hbtxr_config.hpp"
#include "hbtxr_golden.hpp"
#include "hbtxr_model.hpp"
#include "hbtxr_top.hpp"

using namespace hbtxr;
typedef HbtxrCfgBase CFG;
typedef HbtxrCfgHead HCFG;
template struct hbtxr::HbtxrCfgCheck<CFG>;

static constexpr int D = CFG::D, ANCHOR = 5, OUT_DIM = 5;
typedef HbtxrTop<CFG> Top;
static Top top;
static HbtxrModelWeights<CFG> *weights;
static std::vector<double> last_fixed;

/// Compares the stages that have a golden, so a mismatch reports where it started.
struct ModelTrace {
  std::string path;
  std::vector<long long> block_want;
  int tokens = 0, bad = 0, checked = 0;
  const std::string dir;
  explicit ModelTrace(const std::string &d) : dir(d) {}

  template <class T>
  void cmp(const char *name, const T *got, int n, const std::vector<long long> &want) {
    ++checked;
    for (int k = 0; k < n; ++k)
      if ((long long)got[k] != want[k]) {
        std::printf("  %-10s FAIL  first at [%d]: %lld vs %lld\n", name, k, (long long)got[k],
                    want[k]);
        ++bad;
        return;
      }
    std::printf("  %-10s ok    %d values\n", name, n);
  }
  template <class T>
  void stage(const char *name, const T *got, int n) {
    if (!std::string("stem").compare(name)) cmp(name, got, n, read_golden(dir, "stem_" + path + "_y"));
    else if (!std::string("fnorm").compare(name)) cmp(name, got, n, read_golden(dir, path + "_fnorm_y"));
    else if (!std::string("pooled").compare(name)) cmp(name, got, n, read_golden(dir, path + "_pooled"));
    else if (!std::string("head_x").compare(name)) cmp(name, got, n, read_golden(dir, path + "_head_x"));
  }
  void block(int i, const CFG::act_t *y, int n) {
    ++checked;
    for (int k = 0; k < n * D; ++k)
      if ((long long)y[k] != block_want[(size_t)i * n * D + k]) {
        std::printf("  TRB %-2d     FAIL  first at [%d]\n", i, k);
        ++bad;
        return;
      }
  }
};

template <int IN>
static void load_head(HbtxrHead<HCFG, IN, OUT_DIM> &h, const std::string &dir,
                      const std::string &p) {
  const std::vector<long long> w1 = read_golden(dir, p + "_head_fc1_weight");
  const std::vector<long long> b1 = read_golden(dir, p + "_head_fc1_bias_acc");
  const std::vector<long long> w2 = read_golden(dir, p + "_head_fc2_weight");
  const std::vector<long long> b2 = read_golden(dir, p + "_head_fc2_bias_acc");
  const std::vector<long long> gs = read_golden(dir, p + "_head_gelu_scalars");
  const std::vector<long long> gt = read_golden(dir, p + "_head_gelu_table");
  std::vector<long long> m, s;
  read_requant<CFG>(dir, p + "_head_e_fc1_acc", m, s);
  for (int o = 0; o < IN; ++o) {
    for (int i = 0; i < IN; ++i) h.fc1_w[o][i] = w1[(long long)o * IN + i];
    h.fc1_b[o] = b1[o];
    h.fc1_m[o] = m[o];
    h.fc1_s[o] = (int)s[o];
  }
  for (int o = 0; o < OUT_DIM; ++o) {
    for (int i = 0; i < IN; ++i) h.fc2_w[o][i] = w2[(long long)o * IN + i];
    h.fc2_b[o] = b2[o];
  }
  h.gelu_b = (int)gs[0];
  h.gelu_s = (int)gs[1];
  h.gelu_bound = (int)gs[2];
  for (int i = 0; i < 32; ++i) h.gelu_table[i] = gt[i];
  std::vector<long long> gm, gsh;
  read_requant<CFG>(dir, p + "_head_e_gelu_to_fc2", gm, gsh);
  h.e_gelu_m = gm[0];
  h.e_gelu_s = (int)gsh[0];
}

template <class STEM>
static void load_stem(STEM &stem, const std::string &dir, const std::string &p, int cin) {
  const std::vector<long long> w = read_golden(dir, "stem_" + p + "_weight");
  const std::vector<long long> b = read_golden(dir, "stem_" + p + "_bias_acc");
  const std::vector<long long> zp = read_golden(dir, "stem_" + p + "_zp_correction");
  std::vector<long long> m, s;
  read_requant<CFG>(dir, "stem_" + p, m, s);
  constexpr int TAPS = STEM::TAPS;
  static typename HbtxrCfgPatch::w_t wt[D][TAPS];
  static typename HbtxrCfgPatch::acc_t bt[D];
  static ap_uint<CFG::REQ_M_BITS> mt[D];
  static ap_uint<5> st[D];
  for (int o = 0; o < D; ++o) {
    for (int i = 0; i < TAPS; ++i) wt[o][i] = w[(long long)o * TAPS + i];
    bt[o] = b[o] - zp[o];   // the zero-point correction folds into the bias
    mt[o] = m[o];
    st[o] = (int)s[o];
  }
  stem.pe.load(wt, bt, mt, st);
  (void)cin;
}

/// Feed the golden's `[c][y][x]` image as the pixel-interleaved `[y][x][c]` the stem
/// declares. On the board this is the DMA descriptor, not logic.
template <class STEM>
static void push_image(hls::stream<typename STEM::pix_beat_t> &s,
                       const std::vector<long long> &img, int cin, int hin, int win) {
  for (int y = 0; y < hin; ++y)
    for (int x0 = 0; x0 < win * cin; x0 += STEM::LANES) {
      typename STEM::pix_beat_t v;
      for (int l = 0; l < STEM::LANES; ++l) {
        const int idx = x0 + l, x = idx / cin, c = idx % cin;
        v[l] = img[((long long)c * hin + y) * win + x];
      }
      s.write(v);
    }
}

static int run_mode(const std::string &dir, const char *name, int mode) {
  ModelTrace trace(dir);
  trace.path = name;
  trace.block_want = read_golden(dir, std::string(name) + "_block_out");
  const std::vector<long long> meta = read_golden(dir, std::string(name) + "_meta");
  const int cin = (int)meta[0], hin = (int)meta[1], win = (int)meta[2];
  const int anchor_dim = (int)meta[7];
  const std::vector<long long> want = read_golden(dir, std::string(name) + "_head_y_acc");

  hls::stream<Top::stem_f_t::pix_beat_t> frame("frame");
  hls::stream<Top::stem_e_t::pix_beat_t> event("event");
  const std::vector<long long> img = read_golden(dir, "stem_" + std::string(name) + "_image");
  if (mode == 0) push_image<Top::stem_f_t>(frame, img, cin, hin, win);
  else push_image<Top::stem_e_t>(event, img, cin, hin, win);

  HCFG::act_t anchor[ANCHOR] = {};
  if (anchor_dim) {
    const std::vector<long long> a = read_golden(dir, std::string(name) + "_anchor_x");
    for (int i = 0; i < ANCHOR; ++i) anchor[i] = a[i];
  }

  HCFG::acc_t acc[OUT_DIM];
  ap_int<32> fixed[OUT_DIM];
  std::printf("tb_top      %-6s mode=%d  [%d,%d,%d] -> %d TRBs -> %d outputs\n", name, mode,
              cin, hin, win, (int)meta[4], OUT_DIM);
  const int slip = top.run(mode, frame, event, anchor, acc, fixed, *weights, &trace);

  std::vector<long long> got(OUT_DIM);
  for (int o = 0; o < OUT_DIM; ++o) got[o] = (long long)acc[o];
  int bad = compare("head_y_acc", got, want);
  if (slip) { std::printf("  prefetch   FAIL  %d slipped TRBs\n", slip); bad += 1; }
  if (trace.bad) { std::printf("  stages     FAIL  %d of %d\n", trace.bad, trace.checked); bad += 1; }
  else std::printf("  stages     ok    %d checked\n", trace.checked);
  last_fixed.clear();
  std::printf("  fixed Q.%d  ", HBTXR_OUTPUT_FRAC);
  for (int o = 0; o < OUT_DIM; ++o) {
    last_fixed.push_back((double)fixed[o] / (1 << HBTXR_OUTPUT_FRAC));
    std::printf("%.4f ", last_fixed.back());
  }
  std::printf("\n");
  return bad;
}

int main(int argc, char **argv) {
  const std::string dir =
      argc > 1 ? argv[1] : "hardware/workspace/golden/model-hbtxr-w4s4h8";
  weights = new HbtxrModelWeights<CFG>(dir);

  load_stem(top.stem_f, dir, "search", 1);
  load_stem(top.stem_e, dir, "track", 2);
  load_layernorm<CFG>(top.fnorm, dir, "fnorm");
  load_head<D>(top.head_box, dir, "search");
  load_head<D + ANCHOR>(top.head_ellipse, dir, "track");

  const char *paths[2] = {"search", "track"};
  for (int m = 0; m < 2; ++m) {
    top.e_exit_to_fnorm[m] = load_edge<CFG>(dir, std::string(paths[m]) + "_e_exit_to_fnorm");
    top.e_fnorm_to_head[m] = load_edge<CFG>(dir, std::string(paths[m]) + "_e_fnorm_to_head");
  }
  top.e_anchor = load_edge<CFG>(dir, "track_e_anchor");
  for (int m = 0; m < 2; ++m) {
    std::vector<long long> mm, ss;
    read_requant<CFG>(dir, std::string(paths[m]) + "_head_out", mm, ss);
    for (int o = 0; o < OUT_DIM; ++o) { top.out_m[m][o] = mm[o]; top.out_s[m][o] = (int)ss[o]; }
  }

  // search, track, search again. Nothing a mode leaves behind may change the next run of
  // the other — the fixed-point output caught exactly that when its requant table was a
  // single array instead of one per mode.
  int bad = run_mode(dir, "search", 0);
  const std::vector<double> first = last_fixed;
  bad += run_mode(dir, "track", 1);
  bad += run_mode(dir, "search", 0);
  if (first != last_fixed) {
    std::printf("  repeat     FAIL  same input, different result after a mode switch\n");
    bad += 1;
  } else {
    std::printf("  repeat     ok    search reproduces itself across a mode switch\n");
  }

  std::printf("%s\n", bad ? "FAIL" : "PASS");
  return bad;
}
