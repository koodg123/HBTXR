// Stage probe — the ViT_Accel `utils.h` technique (SPEC §9), adapted.
//
// Dropped into a chain between stages it does three things, and the third is the reason
// it exists:
//
//   1. compare each lane against the golden;
//   2. ASSERT THE STREAM IS EMPTY. Over-production and a wrong beat count are invisible
//      to a value comparison — the reference gets this right and it is worth keeping;
//   3. REFILL from the golden, so a fault in stage k does not propagate into k+1 and
//      light up every probe downstream. The first failing probe is then the actual site.
//
// Testbench-only: the core holds a `HbtxrProbe *` that is null in synthesis, so no FIFO
// and no pragma survives into the synthesized path.
#ifndef HBTXR_PROBE_HPP
#define HBTXR_PROBE_HPP

#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include <hls_stream.h>

#include "hbtxr_golden.hpp"

namespace hbtxr {

struct HbtxrProbe {
  std::map<std::string, std::vector<long long>> golden;  // stage name -> expected values
  int failures = 0;
  int checked = 0;
  bool refill = true;   // set false to measure how far a fault actually propagates

  void expect(const std::string &name, std::vector<long long> values) {
    golden[name] = std::move(values);
  }

  /// Register a ROW-MAJOR `[rows, cols]` golden for a stream carrying `TP x P` beats.
  ///
  /// The two orders are not the same and the difference is silent: a beat is
  /// `v[p*P + c]` = row `r0+p`, column `c0+c`, so draining beats in order yields
  /// [row-tile][col-tile][row-in-tile][col-in-tile], not [row][col]. Comparing the raw
  /// drain against a row-major golden mismatches on the ninth value and looks like a
  /// datapath fault.
  void expect2d(const std::string &name, const std::vector<long long> &rowmajor, int rows,
                int cols, int TP, int P) {
    std::vector<long long> out;
    out.reserve((size_t)rows * cols);
    for (int r0 = 0; r0 < rows; r0 += TP)
      for (int c0 = 0; c0 < cols; c0 += P)
        for (int p = 0; p < TP; ++p)
          for (int c = 0; c < P; ++c)
            out.push_back(rowmajor[(size_t)(r0 + p) * cols + c0 + c]);
    golden[name] = std::move(out);
  }

  /// Drain `s`, compare, assert empty, refill. `lanes` is the beat width.
  template <class BEAT>
  void check(const std::string &name, hls::stream<BEAT> &s, int lanes) {
    auto it = golden.find(name);
    if (it == golden.end()) return;             // nothing declared for this stage
    const std::vector<long long> &want = it->second;

    std::vector<long long> got;
    std::vector<BEAT> beats;
    while (!s.empty()) {
      const BEAT b = s.read();
      beats.push_back(b);
      for (int i = 0; i < lanes; ++i) got.push_back((long long)b[i]);
    }
    ++checked;

    if (got.size() != want.size()) {
      std::printf("  probe %-12s FAIL  %zu values, golden has %zu  <-- BEAT COUNT\n",
                  name.c_str(), got.size(), want.size());
      ++failures;
    } else {
      size_t bad = 0, first = 0;
      for (size_t i = 0; i < got.size(); ++i)
        if (got[i] != want[i]) { if (!bad) first = i; ++bad; }
      if (bad) {
        std::printf("  probe %-12s FAIL  first at [%zu]: %lld vs %lld  (%zu/%zu differ)\n",
                    name.c_str(), first, got[first], want[first], bad, got.size());
        ++failures;
      } else {
        std::printf("  probe %-12s ok    %zu values\n", name.c_str(), got.size());
      }
    }

    // Refill. With `refill` off the fault propagates, which is how you see whether a
    // downstream probe found its own bug or merely inherited this one.
    if (refill && got.size() == want.size()) {
      size_t k = 0;
      for (BEAT &b : beats) {
        for (int i = 0; i < lanes; ++i) b[i] = want[k++];
        s.write(b);
      }
    } else {
      for (const BEAT &b : beats) s.write(b);
    }
  }
};

}  // namespace hbtxr

#endif  // HBTXR_PROBE_HPP
