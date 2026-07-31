// Testbench-side reader for the integer goldens (`hardware/tools/export_hls_golden.py`).
//
// Testbench only — never included by anything that gets synthesized.
#ifndef HBTXR_GOLDEN_HPP
#define HBTXR_GOLDEN_HPP

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace hbtxr {

/// One emitted `.txt`: decimal integers, comma separated, TRAILING COMMA, no newline.
/// The trailing comma is why a naive split yields an empty last field.
inline std::vector<long long> read_golden(const std::string &dir, const std::string &name) {
  const std::string path = dir + "/" + name + ".txt";
  FILE *f = std::fopen(path.c_str(), "rb");
  if (!f) {
    std::fprintf(stderr, "cannot open %s\n  run: sh hardware/build/make_golden.sh\n",
                 path.c_str());
    std::exit(2);
  }
  std::vector<long long> out;
  long long v = 0;
  int sign = 1, digits = 0;
  for (int c = std::fgetc(f); c != EOF; c = std::fgetc(f)) {
    if (c == '-') { sign = -1; }
    else if (c >= '0' && c <= '9') { v = v * 10 + (c - '0'); ++digits; }
    else if (c == ',') { if (digits) out.push_back(sign * v); v = 0; sign = 1; digits = 0; }
  }
  if (digits) out.push_back(sign * v);
  std::fclose(f);
  return out;
}

/// Load a `(mult, shift)` pair and REJECT anything the config cannot represent.
///
/// `REQ_M_BITS` is not a property of `dyadic_params` — it is a property of the ratios the
/// design actually presents. A ratio of 1.15 needs 33 bits at shift 31; a ratio of 4.5
/// needs 34. Without this check an oversized multiplier is silently truncated by the
/// `ap_uint<REQ_M_BITS>` parameter and the requant quietly computes something else.
template <class CFG>
inline void read_requant(const std::string &dir, const std::string &edge,
                         std::vector<long long> &mult, std::vector<long long> &shift) {
  mult = read_golden(dir, edge + "_mult");
  shift = read_golden(dir, edge + "_shift");
  if (mult.size() != shift.size()) {
    std::fprintf(stderr, "%s: %zu multipliers but %zu shifts\n", edge.c_str(), mult.size(),
                 shift.size());
    std::exit(2);
  }
  for (size_t i = 0; i < mult.size(); ++i) {
    if (mult[i] <= 0 || mult[i] >= (1LL << CFG::REQ_M_BITS)) {
      std::fprintf(stderr, "%s[%zu]: multiplier %lld does not fit REQ_M_BITS=%d\n",
                   edge.c_str(), i, mult[i], CFG::REQ_M_BITS);
      std::exit(2);
    }
    if (shift[i] < 0 || shift[i] > CFG::REQ_N_MAX) {
      std::fprintf(stderr, "%s[%zu]: shift %lld exceeds REQ_N_MAX=%d\n", edge.c_str(), i,
                   shift[i], CFG::REQ_N_MAX);
      std::exit(2);
    }
  }
}

/// Compare element-wise. No tolerance — the golden is arbitrary-precision integer.
/// Reports the FIRST mismatch, which is the one that locates the bug.
inline int compare(const char *what, const std::vector<long long> &got,
                   const std::vector<long long> &want) {
  if (got.size() != want.size()) {
    std::printf("  %-10s FAIL  %zu values, golden has %zu\n", what, got.size(), want.size());
    return 1;
  }
  for (size_t i = 0; i < got.size(); ++i) {
    if (got[i] != want[i]) {
      size_t n = 0;
      for (size_t j = 0; j < got.size(); ++j) n += (got[j] != want[j]);
      std::printf("  %-10s FAIL  first at [%zu]: %lld vs %lld  (%zu/%zu differ)\n",
                  what, i, got[i], want[i], n, got.size());
      return 1;
    }
  }
  std::printf("  %-10s ok    %zu values, element-wise ==\n", what, got.size());
  return 0;
}

/// The negative control SPEC §9 demands: a deliberately wrong result MUST fail, or the
/// pass above measured nothing. Returns 0 when the comparison correctly rejects.
inline int expect_reject(const char *what, std::vector<long long> got,
                         const std::vector<long long> &want) {
  if (got.empty()) { std::printf("  %-10s FAIL  nothing to perturb\n", what); return 1; }
  got[got.size() / 2] += 1;
  for (size_t i = 0; i < got.size(); ++i) {
    if (got[i] != want[i]) { std::printf("  %-10s ok    a wrong value is rejected\n", what);
                             return 0; }
  }
  std::printf("  %-10s FAIL  a perturbed vector still compares equal\n", what);
  return 1;
}

}  // namespace hbtxr

#endif  // HBTXR_GOLDEN_HPP
