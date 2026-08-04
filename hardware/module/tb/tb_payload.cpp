// V1 — the host/kernel payload contract (SPEC §8).
//
//   python3 hardware/deploy/hbtxr/payload.py --golden <dir>
//   ./tb_payload <golden-dir> <hbtxr_weights.bin>
//
// Loads a core pair TWICE — once from the golden `.txt` through `HbtxrModelWeights`, once
// from the blob through `HbtxrBlobSource` — and compares every member element-wise. Both
// paths end at the same `rmu.load(...)` and the same member assignments, so anything that
// differs is the packing, and the label says which section.
//
// This is a stronger test than running the model on blob weights would be. An end-to-end
// run says "the pupil moved"; this says "b3.fc1.rq disagrees at channel 417". The model is
// already proved against the golden by tb_top -- what was unproved is the transport.
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

#include "hbtxr_config.hpp"
#include "hbtxr_golden.hpp"
#include "hbtxr_model.hpp"
#include "hbtxr_payload.hpp"

using namespace hbtxr;
typedef HbtxrCfgBase CFG;

static std::vector<unsigned char> read_file(const char *path) {
  FILE *f = std::fopen(path, "rb");
  if (!f) {
    std::fprintf(stderr, "cannot open %s\n", path);
    std::fprintf(stderr, "  run: python3 hardware/deploy/hbtxr/payload.py --golden <dir>\n");
    std::exit(2);
  }
  std::fseek(f, 0, SEEK_END);
  const long n = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  std::vector<unsigned char> out((size_t)n);
  if (n > 0 && std::fread(&out[0], 1, (size_t)n, f) != (size_t)n) {
    std::fprintf(stderr, "short read on %s\n", path);
    std::exit(2);
  }
  std::fclose(f);
  return out;
}

typedef std::pair<std::string, std::vector<long long> > Group;

template <int CO, int CI, class RMU>
static void harvest_rmu(const RMU &r, const std::string &p, std::vector<Group> &g) {
  std::vector<long long> w, b, m, s;
  w.reserve((size_t)CO * CI);
  for (int o = 0; o < CO; ++o) {
    for (int i = 0; i < CI; ++i) w.push_back((long long)r.weight[o][i]);
    b.push_back((long long)r.bias[o]);
    m.push_back((long long)r.mult[o]);
    s.push_back((long long)r.shift[o]);
  }
  g.push_back(Group(p + ".w", w));
  g.push_back(Group(p + ".b", b));
  g.push_back(Group(p + ".mult", m));
  g.push_back(Group(p + ".shift", s));
}

template <class LN>
static void harvest_ln(const LN &ln, const std::string &p, std::vector<Group> &g) {
  std::vector<long long> sc, w, b, rs;
  sc.push_back((long long)ln.c_1_m); sc.push_back(ln.c_1_s); sc.push_back(ln.b);
  sc.push_back(ln.s1); sc.push_back(ln.bound); sc.push_back(ln.s2);
  sc.push_back(ln.clamp_bits);
  for (int i = 0; i < CFG::D; ++i) {
    w.push_back((long long)ln.lnw[i]);
    b.push_back((long long)ln.lnb[i]);
  }
  for (int i = 0; i < 64; ++i) rs.push_back((long long)ln.rsqrt[i]);
  g.push_back(Group(p + ".sc", sc));
  g.push_back(Group(p + ".lnw", w));
  g.push_back(Group(p + ".lnb", b));
  g.push_back(Group(p + ".rsqrt", rs));
}

static void harvest_edges(const HbtxrEdge *e, int n, const std::string &p,
                          std::vector<Group> &g) {
  std::vector<long long> v;
  for (int i = 0; i < n; ++i) {
    v.push_back(e[i].mult);
    v.push_back(e[i].shift);
  }
  g.push_back(Group(p, v));
}

static void harvest(const HbtxrMhaCore<CFG> &m, const HbtxrMlpCore<CFG> &p,
                    std::vector<Group> &g) {
  harvest_ln(m.ln, "ln1", g);
  harvest_ln(p.ln, "ln2", g);

  std::vector<long long> sm, ex, r1, r2, gs, gt;
  sm.push_back(m.softmax.b1); sm.push_back(m.softmax.s1); sm.push_back(m.softmax.bound1);
  sm.push_back(m.softmax.b2_one); sm.push_back(m.softmax.s2_one);
  sm.push_back(m.softmax.bound2_one); sm.push_back(m.softmax.b3_one);
  sm.push_back(m.softmax.s3_one); sm.push_back(m.softmax.b2_two);
  sm.push_back(m.softmax.s2_two); sm.push_back(m.softmax.bound2_two);
  sm.push_back(m.softmax.b3_two); sm.push_back(m.softmax.s3_two);
  sm.push_back(m.softmax.clamp_bits);
  for (int i = 0; i < 32; ++i) ex.push_back((long long)m.softmax.exp_table[i]);
  for (int i = 0; i < 64; ++i) {
    r1.push_back((long long)m.softmax.recip_one[i]);
    r2.push_back((long long)m.softmax.recip_two[i]);
  }
  gs.push_back(p.gelu_b); gs.push_back(p.gelu_s); gs.push_back(p.gelu_bound);
  for (int i = 0; i < 32; ++i) gt.push_back((long long)p.gelu_table[i]);
  g.push_back(Group("sm.sc", sm));
  g.push_back(Group("sm.exp", ex));
  g.push_back(Group("sm.r1", r1));
  g.push_back(Group("sm.r2", r2));
  g.push_back(Group("gelu.sc", gs));
  g.push_back(Group("gelu.t", gt));

  harvest_rmu<3 * CFG::D, CFG::D>(m.qkv, "qkv", g);
  harvest_rmu<CFG::D, CFG::D>(m.proj, "proj", g);
  harvest_rmu<CFG::F, CFG::D>(p.fc1, "fc1", g);
  harvest_rmu<CFG::D, CFG::F>(p.fc2, "fc2", g);

  const HbtxrEdge me[7] = {m.e01_stream_to_ln1, m.e02_ln1_to_qkv, m.e04_smu_to_softmax,
                           m.e05_softmax_to_av, m.e06_av_to_proj, m.e08_resid1_a,
                           m.e08_resid1_b};
  const HbtxrEdge pe[5] = {p.e09_stream_to_ln2, p.e10_ln2_to_fc1, p.e12_gelu_to_fc2,
                           p.e14_resid2_a, p.e14_resid2_b};
  harvest_edges(me, 7, "edges.mha", g);
  harvest_edges(pe, 5, "edges.mlp", g);
}

/// Silent on success -- 8 blocks x 26 groups is 208 lines of "ok" that nobody reads, and
/// the point of the group split is that a FAILURE names its section.
static int cmp_quiet(int blk, const Group &got, const Group &want) {
  if (got.first != want.first) {
    std::printf("  b%d %-12s FAIL  harvest order differs: %s vs %s\n", blk, got.first.c_str(),
                got.first.c_str(), want.first.c_str());
    return 1;
  }
  if (got.second.size() != want.second.size()) {
    std::printf("  b%d %-12s FAIL  %zu values, golden has %zu\n", blk, got.first.c_str(),
                got.second.size(), want.second.size());
    return 1;
  }
  for (size_t i = 0; i < got.second.size(); ++i) {
    if (got.second[i] != want.second[i]) {
      size_t n = 0;
      for (size_t j = 0; j < got.second.size(); ++j)
        n += (got.second[j] != want.second[j]);
      std::printf("  b%d %-12s FAIL  first at [%zu]: %lld vs %lld  (%zu/%zu differ)\n", blk,
                  got.first.c_str(), i, got.second[i], want.second[i], n,
                  got.second.size());
      return 1;
    }
  }
  return 0;
}

static std::vector<long long> gv(const std::string &dir, const std::string &name) {
  return read_golden(dir, name);
}

/// Decode a section at its declared width and compare against the golden it came from.
static int check_flat(const HbtxrBlobView &v, const char *name, int bits,
                      const std::vector<long long> &want) {
  const HbtxrSection s = v.find(name);
  if (!s.ok) { std::printf("  %-18s FAIL  section absent\n", name); return 1; }
  if (s.bits != bits || s.count != (int)want.size()) {
    std::printf("  %-18s FAIL  %d x %db, golden is %zu x %db\n", name, s.count, s.bits,
                want.size(), bits);
    return 1;
  }
  for (int i = 0; i < s.count; ++i) {
    long long got = 0;
    switch (bits) {
      case 4:  got = v.i4(s, i); break;
      case 8:  got = v.i8(s, i); break;
      case 16: got = v.i16(s, i); break;
      case 32: got = v.i32(s, i); break;
      case 64: got = v.i64(s, i); break;
      default: std::printf("  %-18s FAIL  no reader for %d bits\n", name, bits); return 1;
    }
    if (got != want[i]) {
      std::printf("  %-18s FAIL  [%d]: %lld vs %lld\n", name, i, got, want[i]);
      return 1;
    }
  }
  return 0;
}

/// A `(mult, shift)` section against the golden's two files. They are packed into one
/// word because they are one decision; unpacking has to put them back the same way.
static int check_rq(const HbtxrBlobView &v, const char *name, const std::string &dir,
                    const std::string &edge) {
  const std::vector<long long> m = read_golden(dir, edge + "_mult");
  const std::vector<long long> sh = read_golden(dir, edge + "_shift");
  const HbtxrSection s = v.find(name);
  if (!s.ok || s.count != (int)m.size() || s.bits != 32) {
    std::printf("  %-18s FAIL  %d entries, golden has %zu\n", name, s.ok ? s.count : -1,
                m.size());
    return 1;
  }
  for (int i = 0; i < s.count; ++i) {
    long long gm = 0;
    int gs = 0;
    v.requant(s, i, gm, gs);
    if (gm != m[i] || gs != (int)sh[i]) {
      std::printf("  %-18s FAIL  [%d]: (%lld,%d) vs (%lld,%lld)\n", name, i, gm, gs, m[i],
                  sh[i]);
      return 1;
    }
  }
  return 0;
}

// Two core pairs, allocated once. Each is ~1 MB of resident weights, and a fresh pair per
// block would be 16 allocations of it.
static HbtxrMhaCore<CFG> g_mha, b_mha;
static HbtxrMlpCore<CFG> g_mlp, b_mlp;

int main(int argc, char **argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <golden-dir> <hbtxr_weights.bin>\n", argv[0]);
    return 2;
  }
  const std::string dir = argv[1];
  const std::vector<unsigned char> raw = read_file(argv[2]);

  HbtxrBlobSource<CFG> blob;
  if (!blob.open(&raw[0], raw.size())) {
    std::printf("tb_payload  FAIL  %s is not an HBTXR payload blob (%zu bytes)\n", argv[2],
                raw.size());
    return 1;
  }
  std::printf("tb_payload  %zu bytes, %d sections, D=%d F=%d depth=%d\n", raw.size(),
              blob.v.n, CFG::D, CFG::F, CFG::SEARCH_DEPTH);

  HbtxrModelWeights<CFG> gold(dir);

  int fail = 0, groups = 0;
  for (int blk = 0; blk < CFG::SEARCH_DEPTH; ++blk) {
    gold.load(blk, g_mha, g_mlp);
    blob.load(blk, b_mha, b_mlp);

    std::vector<Group> want, got;
    harvest(g_mha, g_mlp, want);
    harvest(b_mha, b_mlp, got);
    if (want.size() != got.size()) {
      std::printf("  b%d FAIL  %zu groups vs %zu\n", blk, got.size(), want.size());
      return 1;
    }
    int bad = 0;
    for (size_t i = 0; i < want.size(); ++i) bad += cmp_quiet(blk, got[i], want[i]);
    groups += (int)want.size();
    fail += bad;
    if (!bad) std::printf("  b%d         ok    %zu groups, element-wise ==\n", blk,
                          want.size());
  }

  if (blob.errors) {
    std::printf("  sections   FAIL  %d absent or wrong-shaped, first: %s\n", blob.errors,
                blob.missing);
    ++fail;
  } else {
    std::printf("  sections   ok    every section present at the declared count and width\n");
  }

  // --- stem, terminal decode and the mode table ---------------------------------
  // The blocks above are 96% of the blob and all of the SOURCE contract. What is left is
  // the 4% this project has repeatedly found bugs in: the two stems, the SHARED final
  // norm, the two heads, and the four things that depend on the mode. Compared straight
  // against the golden rather than through a second loader -- there is no
  // `HbtxrModelWeights` equivalent for these, `tb_top` reads them inline.
  fail += check_flat(blob.v, "stem.search.w", 8, gv(dir, "stem_search_weight"));
  fail += check_flat(blob.v, "stem.track.w", 8, gv(dir, "stem_track_weight"));
  for (int i = 0; i < 2; ++i) {
    const char *m = i ? "track" : "search";
    // The zero-point correction is FOLDED into the bias here, not carried separately:
    // it is per out-channel and constant, so it belongs on the accumulator the RMU
    // already adds to (hbtxr_patch_embed.hpp). The check has to fold it too.
    std::vector<long long> b = gv(dir, std::string("stem_") + m + "_bias_acc");
    const std::vector<long long> z = gv(dir, std::string("stem_") + m + "_zp_correction");
    for (size_t k = 0; k < b.size(); ++k) b[k] += z[k];
    fail += check_flat(blob.v, (std::string("stem.") + m + ".b").c_str(), 32, b);
    fail += check_rq(blob.v, (std::string("stem.") + m + ".rq").c_str(), dir,
                     std::string("stem_") + m);
  }

  fail += check_flat(blob.v, "fnorm.sc", 32, gv(dir, "fnorm_scalars"));
  fail += check_flat(blob.v, "fnorm.w", 16, gv(dir, "fnorm_lnw"));
  fail += check_flat(blob.v, "fnorm.b", 64, gv(dir, "fnorm_lnb"));
  fail += check_flat(blob.v, "fnorm.rs", 16, gv(dir, "fnorm_rsqrt_table"));

  for (int i = 0; i < 2; ++i) {
    const std::string m = i ? "track" : "search";
    const std::string h = "head." + m + ".";
    fail += check_flat(blob.v, (h + "fc1w").c_str(), 8, gv(dir, m + "_head_fc1_weight"));
    fail += check_flat(blob.v, (h + "fc1b").c_str(), 32, gv(dir, m + "_head_fc1_bias_acc"));
    fail += check_rq(blob.v, (h + "fc1rq").c_str(), dir, m + "_head_e_fc1_acc");
    fail += check_flat(blob.v, (h + "gsc").c_str(), 32, gv(dir, m + "_head_gelu_scalars"));
    fail += check_flat(blob.v, (h + "gt").c_str(), 16, gv(dir, m + "_head_gelu_table"));
    fail += check_rq(blob.v, (h + "grq").c_str(), dir, m + "_head_e_gelu_to_fc2");
    fail += check_flat(blob.v, (h + "fc2w").c_str(), 8, gv(dir, m + "_head_fc2_weight"));
    fail += check_flat(blob.v, (h + "fc2b").c_str(), 32, gv(dir, m + "_head_fc2_bias_acc"));

    // Exactly four things depend on the mode, and the middle two are the ones that get
    // missed: the requant INTO the one shared norm (the two paths reach it from
    // different producers) and the OUTPUT requant (the two heads' last linears sit on
    // different accumulator grids). A single output table keeps passing the accumulator
    // golden while quietly reporting the wrong fixed-point value for whichever mode ran
    // second -- tb_top's mode-switch check is what caught that.
    const std::string mm = "mode." + m + ".";
    fail += check_rq(blob.v, (mm + "exit").c_str(), dir, m + "_e_exit_to_fnorm");
    fail += check_rq(blob.v, (mm + "feat").c_str(), dir, m + "_e_fnorm_to_head");
    fail += check_rq(blob.v, (mm + "out").c_str(), dir, m + "_head_out");
  }
  fail += check_rq(blob.v, "mode.track.anchor", dir, "track_e_anchor");
  std::printf("  terminal   %s  stems, shared fnorm, both heads, mode table\n",
              fail ? "FAIL" : "ok  ");

  // The negative control SPEC §9 demands. Corrupt one nibble of b0's fc1 weights in a COPY
  // of the blob and confirm the comparison rejects it -- without this the pass above only
  // shows that two readers agree on something, not that they agree on the right thing.
  {
    std::vector<unsigned char> bad = raw;
    HbtxrBlobSource<CFG> probe;
    probe.open(&bad[0], bad.size());
    const HbtxrSection s = probe.v.find("b0.fc1.w");
    if (!s.ok) {
      std::printf("  negative   FAIL  b0.fc1.w not found to perturb\n");
      ++fail;
    } else {
      bad[s.off + s.bytes / 2] ^= 0x10;    // one nibble, deep inside the tensor
      HbtxrBlobSource<CFG> pb;
      pb.open(&bad[0], bad.size());
      gold.load(0, g_mha, g_mlp);
      pb.load(0, b_mha, b_mlp);
      std::vector<Group> want, got;
      harvest(g_mha, g_mlp, want);
      harvest(b_mha, b_mlp, got);
      bool differs = false;
      for (size_t i = 0; i < want.size() && !differs; ++i)
        differs = (got[i].second != want[i].second);
      if (differs) {
        std::printf("  negative   ok    a single flipped nibble is rejected\n");
      } else {
        std::printf("  negative   FAIL  a corrupted blob still compares equal\n");
        ++fail;
      }
    }
  }

  if (fail) {
    std::printf("FAIL  %d group(s)\n", fail);
    return 1;
  }
  std::printf("  loads      ok    %d golden, %d blob\n", gold.loads, blob.loads);
  std::printf("PASS  %d groups over %d blocks\n", groups, CFG::SEARCH_DEPTH);
  return 0;
}
