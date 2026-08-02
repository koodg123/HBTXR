// hbtxr_top — the whole accelerator: stem, shared backbone, terminal decode (SPEC §7, §8).
//
//   mode 0  search:  Conv-F(1,128²) -> B_1:8 -> norm -> pool -> PupilBox(192->192->5)
//   mode 1  track:   Conv-E(2, 64²) -> B_1:4 -> norm -> pool -> PupilEllipse(197->197->5)
//                                      \____ the SAME weights and the SAME norm ____/
//
// ONE design routes both. The old implementation split into `search_profile_top` and
// `track_profile_top`, which is not the paper's structure.
//
// The mode-dependent parts are exactly three, and the middle one is easy to miss:
//   - which stem runs;
//   - the requant INTO the shared final norm. Track leaves block 3 and search leaves
//     block 7, on different grids, and the norm has one input port;
//   - which head runs, and whether the host's anchor state is concatenated.
#ifndef HBTXR_TOP_HPP
#define HBTXR_TOP_HPP

#include <ap_int.h>
#include <hls_stream.h>

#include "hbtxr_backbone.hpp"
#include "hbtxr_config.hpp"
#include "hbtxr_head.hpp"
#include "hbtxr_patch_embed.hpp"

namespace hbtxr {

/// The host-visible result: the head's accumulator requantized onto a fixed-point grid.
/// `OUTPUT_FRAC` matches the generator's, so `y * 2^-FRAC` is the real value.
static constexpr int HBTXR_OUTPUT_FRAC = 16;

template <class CFG>
struct HbtxrTop {
  static constexpr int D = CFG::D, N = CFG::N, TP = CFG::TP, P = CFG::NL_P;
  static constexpr int ANCHOR = 5, OUT_DIM = 5;
  typedef HbtxrCfgHead HCFG;
  typedef typename CFG::act_t act_t;
  typedef typename HCFG::act_t head_t;
  typedef typename HCFG::acc_t hacc_t;

  typedef HbtxrPatchEmbed<HbtxrCfgPatch, CFG, 1, 16, 128, 128, P> stem_f_t;
  typedef HbtxrPatchEmbed<HbtxrCfgPatch, CFG, 2, 16, 64, 64, P> stem_e_t;
  typedef HbtxrLayerNorm<CFG, D, P, head_t> fnorm_t;

  stem_f_t stem_f;                 // Conv-F, frame
  stem_e_t stem_e;                 // Conv-E, event
  HbtxrBackbone<CFG> backbone;
  fnorm_t fnorm;                   // shared by both paths
  HbtxrHead<HCFG, D, OUT_DIM> head_box;              // search, Pupil Box
  HbtxrHead<HCFG, D + ANCHOR, OUT_DIM> head_ellipse; // track, Pupil Ellipse

  // Mode-dependent requants, indexed by mode.
  HbtxrEdge e_exit_to_fnorm[2], e_fnorm_to_head[2], e_anchor;
  // The output requant is mode-dependent TOO: the two heads' last linears have different
  // accumulator grids. A single table here reads correct — the accumulator golden still
  // matches, because the accumulator is upstream of it — and silently reports the wrong
  // fixed-point value for whichever mode ran second.
  ap_uint<CFG::REQ_M_BITS> out_m[2][OUT_DIM];
  ap_uint<5> out_s[2][OUT_DIM];

  act_t tokens_buf[N][D];
  head_t normed[N * D];
  head_t feat[D + ANCHOR];

  /// `anchor` is the host's 5-dim pupil state and is read only in track mode.
  template <class SOURCE, class TRACE>
  int run(int mode, hls::stream<typename stem_f_t::pix_beat_t> &frame,
          hls::stream<typename stem_e_t::pix_beat_t> &event, const head_t *anchor,
          hacc_t *acc_out, ap_int<32> *fixed_out, SOURCE &src, TRACE *trace) {
    const HbtxrSchedule sched = hbtxr_schedule<CFG>(mode);
    const int tokens = sched.tokens;

    // --- stem: modality-specific, one token protocol -------------------------
    {
      hls::stream<typename stem_f_t::tok_beat_t> tok("tok");
      if (mode == 0) stem_f.run(frame, tok);
      else stem_e.run(event, tok);
      HbtxrMhaCore<CFG>::pop2d(tok, &tokens_buf[0][0], tokens, D);
    }
    if (trace) trace->stage("stem", &tokens_buf[0][0], tokens * D);

    // --- the shared backbone --------------------------------------------------
    static act_t stream_out[N * D];
    const int tag_slip =
        backbone.run(&tokens_buf[0][0], stream_out, sched, src, (HbtxrNoProbe *)nullptr,
                     trace);

    // --- terminal decode ------------------------------------------------------
    // The exit bridge is mode-dependent: the two paths reach this one norm from
    // different producers, hence different grids.
    {
      hls::stream<typename fnorm_t::in_beat_t> fin("fin");
      hls::stream<typename fnorm_t::beat_t> fout("fout");
      hls::stream<typename fnorm_t::in_beat_t> raw("raw");
      HbtxrMhaCore<CFG>::push2d(raw, stream_out, tokens, D);
      requant_stream<CFG, typename fnorm_t::in_beat_t, typename fnorm_t::in_beat_t,
                     typename CFG::acc_t>(raw, fin, (tokens / TP) * (D / P), TP * P,
                                          e_exit_to_fnorm[mode].mult,
                                          e_exit_to_fnorm[mode].shift);
      fnorm.run(fin, fout, tokens);
      HbtxrMhaCore<CFG>::pop2d(fout, normed, tokens, D);
    }
    if (trace) trace->stage("fnorm", normed, tokens * D);

    head_t pooled[D];
    hbtxr_pool<HCFG, D, head_t>(normed, pooled, tokens);
    if (trace) trace->stage("pooled", pooled, D);

    for (int i = 0; i < D; ++i)
      feat[i] = requant<HCFG, head_t>(hacc_t(pooled[i]), e_fnorm_to_head[mode].mult,
                                      e_fnorm_to_head[mode].shift);

    if (mode == 0) {
      head_box.run(feat, acc_out);
    } else {
      // The anchor arrives on the HOST's grid and has to be requantized onto the pooled
      // feature's before the concat: a Linear has one input grid.
      for (int i = 0; i < ANCHOR; ++i)
        feat[D + i] = requant<HCFG, head_t>(hacc_t(anchor[i]), e_anchor.mult, e_anchor.shift);
      head_ellipse.run(feat, acc_out);
    }
    if (trace) trace->stage("head_x", feat, mode == 0 ? D : D + ANCHOR);

    // The accumulator is what the golden holds; the fixed-point form is what the AXIS
    // port carries, so the host multiplies nothing.
    for (int o = 0; o < OUT_DIM; ++o)
      fixed_out[o] = requant<HCFG, ap_int<32>>(acc_out[o], out_m[mode][o], out_s[mode][o]);
    return tag_slip;
  }
};

}  // namespace hbtxr

#endif  // HBTXR_TOP_HPP
