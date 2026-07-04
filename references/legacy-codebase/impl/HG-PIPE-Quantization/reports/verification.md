# HG-PIPE Quantization Verification Report

- Source: `/home/user/project/PRJXR/impl_repos/HGPIPE/ICCAD24-HG-PIPE`
- Cases: 97/97 passed
- Elements checked: 5899008
- Total mismatches: 0

## Case Summary

| Kind | Cases | Passed | Elements | Mismatches |
|---|---:|---:|---:|---:|
| gelu_requant_table | 12 | 12 | 1806336 | 0 |
| layernorm_rsqrt_table | 25 | 25 | 903360 | 0 |
| requant_table | 48 | 48 | 1806336 | 0 |
| softmax_segmented_table | 12 | 12 | 1382976 | 0 |

## Cases

| Name | Kind | Elements | Mismatches | IO statistics key |
|---|---|---:|---:|---|
| mlp_0_geluq | gelu_requant_table | 150528 | 0 | mlp0.geluq |
| mlp_10_geluq | gelu_requant_table | 150528 | 0 | mlp10.geluq |
| mlp_11_geluq | gelu_requant_table | 150528 | 0 | mlp11.geluq |
| mlp_1_geluq | gelu_requant_table | 150528 | 0 | mlp1.geluq |
| mlp_2_geluq | gelu_requant_table | 150528 | 0 | mlp2.geluq |
| mlp_3_geluq | gelu_requant_table | 150528 | 0 | mlp3.geluq |
| mlp_4_geluq | gelu_requant_table | 150528 | 0 | mlp4.geluq |
| mlp_5_geluq | gelu_requant_table | 150528 | 0 | mlp5.geluq |
| mlp_6_geluq | gelu_requant_table | 150528 | 0 | mlp6.geluq |
| mlp_7_geluq | gelu_requant_table | 150528 | 0 | mlp7.geluq |
| mlp_8_geluq | gelu_requant_table | 150528 | 0 | mlp8.geluq |
| mlp_9_geluq | gelu_requant_table | 150528 | 0 | mlp9.geluq |
| attn_0_lnq | layernorm_rsqrt_table | 37632 | 0 | attn0.lnq |
| attn_10_lnq | layernorm_rsqrt_table | 37632 | 0 | attn10.lnq |
| attn_11_lnq | layernorm_rsqrt_table | 37632 | 0 | attn11.lnq |
| attn_1_lnq | layernorm_rsqrt_table | 37632 | 0 | attn1.lnq |
| attn_2_lnq | layernorm_rsqrt_table | 37632 | 0 | attn2.lnq |
| attn_3_lnq | layernorm_rsqrt_table | 37632 | 0 | attn3.lnq |
| attn_4_lnq | layernorm_rsqrt_table | 37632 | 0 | attn4.lnq |
| attn_5_lnq | layernorm_rsqrt_table | 37632 | 0 | attn5.lnq |
| attn_6_lnq | layernorm_rsqrt_table | 37632 | 0 | attn6.lnq |
| attn_7_lnq | layernorm_rsqrt_table | 37632 | 0 | attn7.lnq |
| attn_8_lnq | layernorm_rsqrt_table | 37632 | 0 | attn8.lnq |
| attn_9_lnq | layernorm_rsqrt_table | 37632 | 0 | attn9.lnq |
| head_lnq | layernorm_rsqrt_table | 192 | 0 | head.lnq |
| mlp_0_lnq | layernorm_rsqrt_table | 37632 | 0 | mlp0.lnq |
| mlp_10_lnq | layernorm_rsqrt_table | 37632 | 0 | mlp10.lnq |
| mlp_11_lnq | layernorm_rsqrt_table | 37632 | 0 | mlp11.lnq |
| mlp_1_lnq | layernorm_rsqrt_table | 37632 | 0 | mlp1.lnq |
| mlp_2_lnq | layernorm_rsqrt_table | 37632 | 0 | mlp2.lnq |
| mlp_3_lnq | layernorm_rsqrt_table | 37632 | 0 | mlp3.lnq |
| mlp_4_lnq | layernorm_rsqrt_table | 37632 | 0 | mlp4.lnq |
| mlp_5_lnq | layernorm_rsqrt_table | 37632 | 0 | mlp5.lnq |
| mlp_6_lnq | layernorm_rsqrt_table | 37632 | 0 | mlp6.lnq |
| mlp_7_lnq | layernorm_rsqrt_table | 37632 | 0 | mlp7.lnq |
| mlp_8_lnq | layernorm_rsqrt_table | 37632 | 0 | mlp8.lnq |
| mlp_9_lnq | layernorm_rsqrt_table | 37632 | 0 | mlp9.lnq |
| attn_0_aq | requant_table | 37632 | 0 | attn0.aq |
| attn_0_kq | requant_table | 37632 | 0 | attn0.kq |
| attn_0_qq | requant_table | 37632 | 0 | attn0.qq |
| attn_0_vq | requant_table | 37632 | 0 | attn0.vq |
| attn_10_aq | requant_table | 37632 | 0 | attn10.aq |
| attn_10_kq | requant_table | 37632 | 0 | attn10.kq |
| attn_10_qq | requant_table | 37632 | 0 | attn10.qq |
| attn_10_vq | requant_table | 37632 | 0 | attn10.vq |
| attn_11_aq | requant_table | 37632 | 0 | attn11.aq |
| attn_11_kq | requant_table | 37632 | 0 | attn11.kq |
| attn_11_qq | requant_table | 37632 | 0 | attn11.qq |
| attn_11_vq | requant_table | 37632 | 0 | attn11.vq |
| attn_1_aq | requant_table | 37632 | 0 | attn1.aq |
| attn_1_kq | requant_table | 37632 | 0 | attn1.kq |
| attn_1_qq | requant_table | 37632 | 0 | attn1.qq |
| attn_1_vq | requant_table | 37632 | 0 | attn1.vq |
| attn_2_aq | requant_table | 37632 | 0 | attn2.aq |
| attn_2_kq | requant_table | 37632 | 0 | attn2.kq |
| attn_2_qq | requant_table | 37632 | 0 | attn2.qq |
| attn_2_vq | requant_table | 37632 | 0 | attn2.vq |
| attn_3_aq | requant_table | 37632 | 0 | attn3.aq |
| attn_3_kq | requant_table | 37632 | 0 | attn3.kq |
| attn_3_qq | requant_table | 37632 | 0 | attn3.qq |
| attn_3_vq | requant_table | 37632 | 0 | attn3.vq |
| attn_4_aq | requant_table | 37632 | 0 | attn4.aq |
| attn_4_kq | requant_table | 37632 | 0 | attn4.kq |
| attn_4_qq | requant_table | 37632 | 0 | attn4.qq |
| attn_4_vq | requant_table | 37632 | 0 | attn4.vq |
| attn_5_aq | requant_table | 37632 | 0 | attn5.aq |
| attn_5_kq | requant_table | 37632 | 0 | attn5.kq |
| attn_5_qq | requant_table | 37632 | 0 | attn5.qq |
| attn_5_vq | requant_table | 37632 | 0 | attn5.vq |
| attn_6_aq | requant_table | 37632 | 0 | attn6.aq |
| attn_6_kq | requant_table | 37632 | 0 | attn6.kq |
| attn_6_qq | requant_table | 37632 | 0 | attn6.qq |
| attn_6_vq | requant_table | 37632 | 0 | attn6.vq |
| attn_7_aq | requant_table | 37632 | 0 | attn7.aq |
| attn_7_kq | requant_table | 37632 | 0 | attn7.kq |
| attn_7_qq | requant_table | 37632 | 0 | attn7.qq |
| attn_7_vq | requant_table | 37632 | 0 | attn7.vq |
| attn_8_aq | requant_table | 37632 | 0 | attn8.aq |
| attn_8_kq | requant_table | 37632 | 0 | attn8.kq |
| attn_8_qq | requant_table | 37632 | 0 | attn8.qq |
| attn_8_vq | requant_table | 37632 | 0 | attn8.vq |
| attn_9_aq | requant_table | 37632 | 0 | attn9.aq |
| attn_9_kq | requant_table | 37632 | 0 | attn9.kq |
| attn_9_qq | requant_table | 37632 | 0 | attn9.qq |
| attn_9_vq | requant_table | 37632 | 0 | attn9.vq |
| attn_0_softmaxq | softmax_segmented_table | 115248 | 0 | attn0.softmaxq |
| attn_10_softmaxq | softmax_segmented_table | 115248 | 0 | attn10.softmaxq |
| attn_11_softmaxq | softmax_segmented_table | 115248 | 0 | attn11.softmaxq |
| attn_1_softmaxq | softmax_segmented_table | 115248 | 0 | attn1.softmaxq |
| attn_2_softmaxq | softmax_segmented_table | 115248 | 0 | attn2.softmaxq |
| attn_3_softmaxq | softmax_segmented_table | 115248 | 0 | attn3.softmaxq |
| attn_4_softmaxq | softmax_segmented_table | 115248 | 0 | attn4.softmaxq |
| attn_5_softmaxq | softmax_segmented_table | 115248 | 0 | attn5.softmaxq |
| attn_6_softmaxq | softmax_segmented_table | 115248 | 0 | attn6.softmaxq |
| attn_7_softmaxq | softmax_segmented_table | 115248 | 0 | attn7.softmaxq |
| attn_8_softmaxq | softmax_segmented_table | 115248 | 0 | attn8.softmaxq |
| attn_9_softmaxq | softmax_segmented_table | 115248 | 0 | attn9.softmaxq |
