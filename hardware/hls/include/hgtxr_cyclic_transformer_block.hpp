#ifndef HGTXR_CYCLIC_TRANSFORMER_BLOCK_HPP
#define HGTXR_CYCLIC_TRANSFORMER_BLOCK_HPP

#include "hgtxr_cyclic_attention.hpp"
#include "hgtxr_cyclic_mlp.hpp"
#include "hgtxr_cyclic_norm.hpp"

namespace hgtxr {
namespace cyclic_transformer {

template <int TM, int TC, int TF>
void qkv_project_tile(const HgtxrDataT x[TM][TC],
                      const HgtxrDataT wq[TC][TC],
                      const HgtxrDataT wk[TC][TC],
                      const HgtxrDataT wv[TC][TC],
                      HgtxrDataT q[TM][TC],
                      HgtxrDataT k[TM][TC],
                      HgtxrDataT v[TM][TC],
                      ap_int<8> proj_scale_shift) {
#pragma HLS INLINE off

  HgtxrAccumT q_acc[TM][TC];
  HgtxrAccumT k_acc[TM][TC];
  HgtxrAccumT v_acc[TM][TC];
#pragma HLS ARRAY_PARTITION variable=q_acc cyclic factor=HGTXR_PAR_FFN dim=2
#pragma HLS ARRAY_PARTITION variable=k_acc cyclic factor=HGTXR_PAR_FFN dim=2
#pragma HLS ARRAY_PARTITION variable=v_acc cyclic factor=HGTXR_PAR_FFN dim=2

  mac_tile<TM, TC, TC>(x, wq, q_acc, false);
  mac_tile<TM, TC, TC>(x, wk, k_acc, false);
  mac_tile<TM, TC, TC>(x, wv, v_acc, false);
  accum_to_data_tile<TM, TC>(q_acc, q, proj_scale_shift);
  accum_to_data_tile<TM, TC>(k_acc, k, proj_scale_shift);
  accum_to_data_tile<TM, TC>(v_acc, v, proj_scale_shift);
}

template <int TM, int TC>
void output_project_tile(const HgtxrDataT x[TM][TC],
                         const HgtxrDataT wo[TC][TC],
                         HgtxrDataT out[TM][TC],
                         ap_int<8> proj_scale_shift) {
#pragma HLS INLINE off
  HgtxrAccumT out_acc[TM][TC];
#pragma HLS ARRAY_PARTITION variable=out_acc cyclic factor=HGTXR_PAR_FFN dim=2
  mac_tile<TM, TC, TC>(x, wo, out_acc, false);
  accum_to_data_tile<TM, TC>(out_acc, out, proj_scale_shift);
}

template <int TM, int TC, int TF>
void transformer_block_tile(const HgtxrDataT x[TM][TC],
                            const HgtxrDataT norm1_gamma[TC],
                            const HgtxrDataT norm1_beta[TC],
                            const HgtxrDataT norm2_gamma[TC],
                            const HgtxrDataT norm2_beta[TC],
                            const HgtxrDataT wq[TC][TC],
                            const HgtxrDataT wk[TC][TC],
                            const HgtxrDataT wv[TC][TC],
                            const HgtxrDataT wo[TC][TC],
                            const HgtxrDataT w1[TC][TF],
                            const HgtxrDataT w2[TF][TC],
                            HgtxrDataT out[TM][TC],
                            ap_int<8> proj_scale_shift,
                            ap_int<8> score_scale_shift,
                            ap_int<8> value_scale_shift,
                            ap_int<8> mlp1_scale_shift,
                            ap_int<8> mlp2_scale_shift,
                            bool enable_norm_mean_center,
                            bool enable_norm_affine) {
#pragma HLS INLINE off

  HgtxrDataT x_norm[TM][TC];
  HgtxrDataT q[TM][TC];
  HgtxrDataT k[TM][TC];
  HgtxrDataT v[TM][TC];
  HgtxrDataT attn[TM][TC];
  HgtxrDataT attn_proj[TM][TC];
  HgtxrDataT resid1[TM][TC];
  HgtxrDataT resid1_norm[TM][TC];

#pragma HLS ARRAY_PARTITION variable=x_norm cyclic factor=HGTXR_PAR_FFN dim=2
#pragma HLS ARRAY_PARTITION variable=q cyclic factor=HGTXR_PAR_FFN dim=2
#pragma HLS ARRAY_PARTITION variable=k cyclic factor=HGTXR_PAR_FFN dim=2
#pragma HLS ARRAY_PARTITION variable=v cyclic factor=HGTXR_PAR_FFN dim=2
#pragma HLS ARRAY_PARTITION variable=attn cyclic factor=HGTXR_PAR_FFN dim=2
#pragma HLS ARRAY_PARTITION variable=attn_proj cyclic factor=HGTXR_PAR_FFN dim=2
#pragma HLS ARRAY_PARTITION variable=resid1 cyclic factor=HGTXR_PAR_FFN dim=2
#pragma HLS ARRAY_PARTITION variable=resid1_norm cyclic factor=HGTXR_PAR_FFN dim=2

  norm_tile<TM, TC>(x, norm1_gamma, norm1_beta, x_norm,
                    enable_norm_mean_center, enable_norm_affine);
  qkv_project_tile<TM, TC, TF>(x_norm, wq, wk, wv, q, k, v, proj_scale_shift);
  attention_tile<TM, TM, TC>(q, k, v, attn, score_scale_shift, value_scale_shift);
  output_project_tile<TM, TC>(attn, wo, attn_proj, proj_scale_shift);
  residual_add_tile<TM, TC>(x, attn_proj, resid1);

  norm_tile<TM, TC>(resid1, norm2_gamma, norm2_beta, resid1_norm,
                    enable_norm_mean_center, enable_norm_affine);
  mlp_tile<TM, TC, TF>(resid1_norm, w1, w2, resid1, out,
                       mlp1_scale_shift, mlp2_scale_shift, true);
}

}  // namespace cyclic_transformer
}  // namespace hgtxr

#endif  // HGTXR_CYCLIC_TRANSFORMER_BLOCK_HPP

