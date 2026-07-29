#!/bin/sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$ROOT_DIR"

PROFILE=${1:-par32_dsp_mixed_stream_mem16}

case "$PROFILE" in
  par16_c3b_mem16)
    PROJECT_NAME="hgtxr_e2e_axis_dma_c3b_mem16_overlay"
    ARTIFACT_NAME="hgtxr_e2e_axis_dma_c3b_mem16"
    HLS_IP_REPO="generated/hgtxr_e2e_axis_par16_c3b_mem16_hls/solution_e2e_q4w8a/impl/ip"
    ;;
  par32_dsp_mixed_stream_mem16)
    PROJECT_NAME="hgtxr_e2e_axis_dma_par32_dsp_mixed_stream_mem16_overlay"
    ARTIFACT_NAME="hgtxr_e2e_axis_dma_par32_dsp_mixed_stream_mem16"
    HLS_IP_REPO="generated/hgtxr_e2e_axis_par32_dsp_mixed_stream_mem16_no_board/solution_e2e_q4w8a/impl/ip"
    ;;
  par32_runtime_mode_mem16)
    PROJECT_NAME="hgtxr_e2e_axis_dma_par32_runtime_mode_mem16_overlay"
    ARTIFACT_NAME="hgtxr_e2e_axis_dma_par32_runtime_mode_mem16"
    HLS_IP_REPO="generated/hgtxr_e2e_axis_par32_runtime_mode_mem16_no_board/solution_e2e_q4w8a/impl/ip"
    ;;
  par32_runtime_full_axi_mem16)
    PROJECT_NAME="hgtxr_e2e_axis_dma_par32_runtime_full_axi_mem16_overlay"
    ARTIFACT_NAME="hgtxr_e2e_axis_dma_par32_runtime_full_axi_mem16"
    HLS_IP_REPO="generated/hgtxr_e2e_axis_par32_runtime_full_axi_mem16_no_board/solution_e2e_q4w8a/impl/ip"
    ;;
  par32_runtime_full_axi_dsp_mem16)
    PROJECT_NAME="hgtxr_e2e_axis_dma_par32_runtime_full_axi_dsp_mem16_overlay"
    ARTIFACT_NAME="hgtxr_e2e_axis_dma_par32_runtime_full_axi_dsp_mem16"
    HLS_IP_REPO="generated/hgtxr_e2e_axis_par32_runtime_full_axi_dsp_mem16_no_board/solution_e2e_q4w8a/impl/ip"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_300_mem16)
    PROJECT_NAME="hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_300_mem16_overlay"
    ARTIFACT_NAME="hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_300_mem16"
    HLS_IP_REPO="generated/hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_300_mem16_no_board/solution_e2e_q4w8a/impl/ip"
    PL_CLK_MHZ="300.0"
    IMPL_STRATEGY="Performance_Explore"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headlut_300_mem16)
    PROJECT_NAME="hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headlut_300_mem16_overlay"
    ARTIFACT_NAME="hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headlut_300_mem16"
    HLS_IP_REPO="generated/hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headlut_300_mem16_no_board/solution_e2e_q4w8a/impl/ip"
    PL_CLK_MHZ="300.0"
    IMPL_STRATEGY="Performance_Explore"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_addrsw_300_mem16)
    PROJECT_NAME="hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_addrsw_300_mem16_overlay"
    ARTIFACT_NAME="hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_addrsw_300_mem16"
    HLS_IP_REPO="generated/hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_addrsw_300_mem16_no_board/solution_e2e_q4w8a/impl/ip"
    PL_CLK_MHZ="300.0"
    IMPL_STRATEGY="Performance_Explore"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_300_mem16)
    PROJECT_NAME="hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_300_mem16_overlay"
    ARTIFACT_NAME="hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_300_mem16"
    HLS_IP_REPO="generated/hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_300_mem16_no_board/solution_e2e_q4w8a/impl/ip"
    PL_CLK_MHZ="300.0"
    IMPL_STRATEGY="Performance_Explore"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_300_mem16)
    PROJECT_NAME="hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_300_mem16_overlay"
    ARTIFACT_NAME="hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_300_mem16"
    HLS_IP_REPO="generated/hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_300_mem16_no_board/solution_e2e_q4w8a/impl/ip"
    PL_CLK_MHZ="300.0"
    IMPL_STRATEGY="Performance_Explore"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_300_mem16)
    PROJECT_NAME="hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_300_mem16_overlay"
    ARTIFACT_NAME="hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_300_mem16"
    HLS_IP_REPO="generated/hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_300_mem16_no_board/solution_e2e_q4w8a/impl/ip"
    PL_CLK_MHZ="300.0"
    IMPL_STRATEGY="Performance_Explore"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_300_mem16)
    PROJECT_NAME="hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_300_mem16_overlay"
    ARTIFACT_NAME="hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_300_mem16"
    HLS_IP_REPO="generated/hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_300_mem16_no_board/solution_e2e_q4w8a/impl/ip"
    PL_CLK_MHZ="300.0"
    IMPL_STRATEGY="Performance_Explore"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_300_mem16)
    PROJECT_NAME="hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_300_mem16_overlay"
    ARTIFACT_NAME="hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_300_mem16"
    HLS_IP_REPO="generated/hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_300_mem16_no_board/solution_e2e_q4w8a/impl/ip"
    PL_CLK_MHZ="300.0"
    IMPL_STRATEGY="Performance_ExplorePostRoutePhysOpt"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_300_mem16)
    PROJECT_NAME="hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_300_mem16_overlay"
    ARTIFACT_NAME="hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_300_mem16"
    HLS_IP_REPO="generated/hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_300_mem16_no_board/solution_e2e_q4w8a/impl/ip"
    PL_CLK_MHZ="300.0"
    IMPL_STRATEGY="Performance_ExplorePostRoutePhysOpt"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_dsppipe4_300_mem16)
    PROJECT_NAME="hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_dsppipe4_300_mem16_overlay"
    ARTIFACT_NAME="hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_dsppipe4_300_mem16"
    HLS_IP_REPO="generated/hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_dsppipe4_300_mem16_no_board/solution_e2e_q4w8a/impl/ip"
    PL_CLK_MHZ="300.0"
    IMPL_STRATEGY="Performance_ExplorePostRoutePhysOpt"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_dsppipe4_300_mem16)
    PROJECT_NAME="hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_dsppipe4_300_mem16_overlay"
    ARTIFACT_NAME="hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_dsppipe4_300_mem16"
    HLS_IP_REPO="generated/hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_dsppipe4_300_mem16_no_board/solution_e2e_q4w8a/impl/ip"
    PL_CLK_MHZ="300.0"
    IMPL_STRATEGY="Performance_ExplorePostRoutePhysOpt"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16)
    PROJECT_NAME="hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_overlay"
    ARTIFACT_NAME="hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16"
    HLS_IP_REPO="generated/hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_no_board/solution_e2e_q4w8a/impl/ip"
    PL_CLK_MHZ="300.0"
    IMPL_STRATEGY="Performance_ExplorePostRoutePhysOpt"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16)
    PROJECT_NAME="hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_overlay"
    ARTIFACT_NAME="hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16"
    HLS_IP_REPO="generated/hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_no_board/solution_e2e_q4w8a/impl/ip"
    PL_CLK_MHZ="300.0"
    IMPL_STRATEGY="Performance_ExplorePostRoutePhysOpt"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16)
    PROJECT_NAME="hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_overlay"
    ARTIFACT_NAME="hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16"
    HLS_IP_REPO="generated/hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_no_board/solution_e2e_q4w8a/impl/ip"
    PL_CLK_MHZ="300.0"
    IMPL_STRATEGY="Performance_ExplorePostRoutePhysOpt"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_tok2_attntok2_attnbram_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16)
    PROJECT_NAME="hgtxr_e2e_axis_dma_par32_rt_attntok2_attnbram_300_mem16_overlay"
    ARTIFACT_NAME="hgtxr_e2e_axis_dma_par32_rt_attntok2_attnbram_300_mem16"
    HLS_IP_REPO="generated/hgtxr_e2e_axis_par32_rt_attntok2_attnbram_300_mem16_no_board/solution_e2e_q4w8a/impl/ip"
    PL_CLK_MHZ="300.0"
    IMPL_STRATEGY="Performance_ExplorePostRoutePhysOpt"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_tok2_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16)
    PROJECT_NAME="hgtxr_e2e_axis_dma_par32_rt_attntok2_attnbram_patch32_300_mem16_overlay"
    ARTIFACT_NAME="hgtxr_e2e_axis_dma_par32_rt_attntok2_attnbram_patch32_300_mem16"
    HLS_IP_REPO="generated/hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_300_mem16_no_board/solution_e2e_q4w8a/impl/ip"
    PL_CLK_MHZ="300.0"
    IMPL_STRATEGY="Performance_ExplorePostRoutePhysOpt"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_mtok2_dtok4_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16)
    PROJECT_NAME="hgtxr_e2e_axis_dma_par32_rt_attntok2_attnbram_patch32_dtok4_300_mem16_overlay"
    ARTIFACT_NAME="hgtxr_e2e_axis_dma_par32_rt_attntok2_attnbram_patch32_dtok4_300_mem16"
    HLS_IP_REPO="generated/hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_dtok4_300_mem16_no_board/solution_e2e_q4w8a/impl/ip"
    PL_CLK_MHZ="300.0"
    IMPL_STRATEGY="Performance_ExplorePostRoutePhysOpt"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_mtok4_dtok4_aq2_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only)
    PROJECT_NAME="hgtxr_e2e_axis_dma_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq2_300_mem16_search_only_overlay"
    ARTIFACT_NAME="hgtxr_e2e_axis_dma_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq2_300_mem16_search_only"
    HLS_IP_REPO="generated/hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq2_300_mem16_search_only_no_board/solution_e2e_q4w8a/impl/ip"
    PL_CLK_MHZ="300.0"
    IMPL_STRATEGY="Performance_ExplorePostRoutePhysOpt"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_mtok4_dtok4_aq2_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_track_only)
    PROJECT_NAME="hgtxr_e2e_axis_dma_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq2_300_mem16_track_only_overlay"
    ARTIFACT_NAME="hgtxr_e2e_axis_dma_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq2_300_mem16_track_only"
    HLS_IP_REPO="generated/hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq2_300_mem16_track_only_no_board/solution_e2e_q4w8a/impl/ip"
    PL_CLK_MHZ="300.0"
    IMPL_STRATEGY="Performance_ExplorePostRoutePhysOpt"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar4_w2hgroup_mtok4_dtok4_aq4_qkvbram_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only)
    PROJECT_NAME="hgtxr_e2e_axis_dma_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq4_hp4_w2hgroup_qkvbram_300_mem16_search_only_overlay"
    ARTIFACT_NAME="hgtxr_e2e_axis_dma_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq4_hp4_w2hgroup_qkvbram_300_mem16_search_only"
    HLS_IP_REPO="generated/hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq4_hp4_w2hgroup_qkvbram_300_mem16_search_only_no_board/solution_e2e_q4w8a/impl/ip"
    PL_CLK_MHZ="300.0"
    IMPL_STRATEGY="Performance_ExplorePostRoutePhysOpt"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar4_w2hgroup_mtok4_dtok4_aq4_qkvbram_tail2_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only)
    PROJECT_NAME="hgtxr_e2e_axis_dma_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq4_hp4_w2hgroup_qkvbram_tail2_300_mem16_search_only_overlay"
    ARTIFACT_NAME="hgtxr_e2e_axis_dma_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq4_hp4_w2hgroup_qkvbram_tail2_300_mem16_search_only"
    HLS_IP_REPO="generated/hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq4_hp4_w2hgroup_qkvbram_tail2_300_mem16_search_only_no_board/solution_e2e_q4w8a/impl/ip"
    PL_CLK_MHZ="300.0"
    IMPL_STRATEGY="Performance_ExplorePostRoutePhysOpt"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_mtok2_dtok4_attntok2_attnbram_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16)
    PROJECT_NAME="hgtxr_e2e_axis_dma_par32_rt_attntok2_attnbram_dtok4_300_mem16_overlay"
    ARTIFACT_NAME="hgtxr_e2e_axis_dma_par32_rt_attntok2_attnbram_dtok4_300_mem16"
    HLS_IP_REPO="generated/hgtxr_e2e_axis_par32_rt_attntok2_attnbram_dtok4_300_mem16_no_board/solution_e2e_q4w8a/impl/ip"
    PL_CLK_MHZ="300.0"
    IMPL_STRATEGY="Performance_ExplorePostRoutePhysOpt"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_toklutbuf_dsppipe4_300_mem16)
    PROJECT_NAME="hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_toklutbuf_dsppipe4_300_mem16_overlay"
    ARTIFACT_NAME="hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_toklutbuf_dsppipe4_300_mem16"
    HLS_IP_REPO="generated/hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_toklutbuf_dsppipe4_300_mem16_no_board/solution_e2e_q4w8a/impl/ip"
    PL_CLK_MHZ="300.0"
    IMPL_STRATEGY="Performance_ExplorePostRoutePhysOpt"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail4_dsppipe4_300_mem16)
    PROJECT_NAME="hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail4_dsppipe4_300_mem16_overlay"
    ARTIFACT_NAME="hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail4_dsppipe4_300_mem16"
    HLS_IP_REPO="generated/hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail4_dsppipe4_300_mem16_no_board/solution_e2e_q4w8a/impl/ip"
    PL_CLK_MHZ="300.0"
    IMPL_STRATEGY="Performance_ExplorePostRoutePhysOpt"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail8_lutbuf_dsppipe4_300_mem16)
    PROJECT_NAME="hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail8_lutbuf_dsppipe4_300_mem16_overlay"
    ARTIFACT_NAME="hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail8_lutbuf_dsppipe4_300_mem16"
    HLS_IP_REPO="generated/hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail8_lutbuf_dsppipe4_300_mem16_no_board/solution_e2e_q4w8a/impl/ip"
    PL_CLK_MHZ="300.0"
    IMPL_STRATEGY="Performance_ExplorePostRoutePhysOpt"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail16_lutbuf_dsppipe4_300_mem16)
    PROJECT_NAME="hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail16_lutbuf_dsppipe4_300_mem16_overlay"
    ARTIFACT_NAME="hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail16_lutbuf_dsppipe4_300_mem16"
    HLS_IP_REPO="generated/hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail16_lutbuf_dsppipe4_300_mem16_no_board/solution_e2e_q4w8a/impl/ip"
    PL_CLK_MHZ="300.0"
    IMPL_STRATEGY="Performance_ExplorePostRoutePhysOpt"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_dsppipe4_keep_300_mem16)
    PROJECT_NAME="hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_dsppipe4_keep_300_mem16_overlay"
    ARTIFACT_NAME="hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_dsppipe4_keep_300_mem16"
    HLS_IP_REPO="generated/hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_dsppipe4_keep_300_mem16_no_board/solution_e2e_q4w8a/impl/ip"
    PL_CLK_MHZ="300.0"
    IMPL_STRATEGY="Performance_ExplorePostRoutePhysOpt"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_300_mem16_search_only)
    PROJECT_NAME="hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_300_mem16_search_only_overlay"
    ARTIFACT_NAME="hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_300_mem16_search_only"
    HLS_IP_REPO="generated/hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_300_mem16_search_only_no_board/solution_e2e_q4w8a/impl/ip"
    PL_CLK_MHZ="300.0"
    IMPL_STRATEGY="Performance_ExplorePostRoutePhysOpt"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_pipepref_300_mem16)
    PROJECT_NAME="hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_pipepref_300_mem16_overlay"
    ARTIFACT_NAME="hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_pipepref_300_mem16"
    HLS_IP_REPO="generated/hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_pipepref_300_mem16_no_board/solution_e2e_q4w8a/impl/ip"
    PL_CLK_MHZ="300.0"
    IMPL_STRATEGY="Performance_Explore"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_pipepref_tail4_nowide_300_mem16)
    PROJECT_NAME="hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_pipepref_tail4_nowide_300_mem16_overlay"
    ARTIFACT_NAME="hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_pipepref_tail4_nowide_300_mem16"
    HLS_IP_REPO="generated/hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_pipepref_tail4_nowide_300_mem16_no_board/solution_e2e_q4w8a/impl/ip"
    PL_CLK_MHZ="300.0"
    IMPL_STRATEGY="Performance_Explore"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_pipepref_300_mem16)
    PROJECT_NAME="hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_pipepref_300_mem16_overlay"
    ARTIFACT_NAME="hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_pipepref_300_mem16"
    HLS_IP_REPO="generated/hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_pipepref_300_mem16_no_board/solution_e2e_q4w8a/impl/ip"
    PL_CLK_MHZ="300.0"
    IMPL_STRATEGY="Performance_Explore"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_pipepref_tail2_300_mem16)
    PROJECT_NAME="hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_pipepref_tail2_300_mem16_overlay"
    ARTIFACT_NAME="hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_pipepref_tail2_300_mem16"
    HLS_IP_REPO="generated/hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_pipepref_tail2_300_mem16_no_board/solution_e2e_q4w8a/impl/ip"
    PL_CLK_MHZ="300.0"
    IMPL_STRATEGY="Performance_Explore"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_pipepref_tail4_300_mem16)
    PROJECT_NAME="hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_pipepref_tail4_300_mem16_overlay"
    ARTIFACT_NAME="hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_pipepref_tail4_300_mem16"
    HLS_IP_REPO="generated/hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_pipepref_tail4_300_mem16_no_board/solution_e2e_q4w8a/impl/ip"
    PL_CLK_MHZ="300.0"
    IMPL_STRATEGY="Performance_Explore"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_pipepref_cttail4_300_mem16)
    PROJECT_NAME="hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_pipepref_cttail4_300_mem16_overlay"
    ARTIFACT_NAME="hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_pipepref_cttail4_300_mem16"
    HLS_IP_REPO="generated/hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_pipepref_cttail4_300_mem16_no_board/solution_e2e_q4w8a/impl/ip"
    PL_CLK_MHZ="300.0"
    IMPL_STRATEGY="Performance_Explore"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_pipepref_tail8_300_mem16)
    PROJECT_NAME="hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_pipepref_tail8_300_mem16_overlay"
    ARTIFACT_NAME="hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_pipepref_tail8_300_mem16"
    HLS_IP_REPO="generated/hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_pipepref_tail8_300_mem16_no_board/solution_e2e_q4w8a/impl/ip"
    PL_CLK_MHZ="300.0"
    IMPL_STRATEGY="Performance_Explore"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_pipepref_coreallfabric_300_mem16)
    PROJECT_NAME="hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_pipepref_coreallfabric_300_mem16_overlay"
    ARTIFACT_NAME="hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_pipepref_coreallfabric_300_mem16"
    HLS_IP_REPO="generated/hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_pipepref_coreallfabric_300_mem16_no_board/solution_e2e_q4w8a/impl/ip"
    PL_CLK_MHZ="300.0"
    IMPL_STRATEGY="Performance_Explore"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_pipepref_coreallfabric_nowide_300_mem16)
    PROJECT_NAME="hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_pipepref_coreallfabric_nowide_300_mem16_overlay"
    ARTIFACT_NAME="hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_pipepref_coreallfabric_nowide_300_mem16"
    HLS_IP_REPO="generated/hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_pipepref_coreallfabric_nowide_300_mem16_no_board/solution_e2e_q4w8a/impl/ip"
    PL_CLK_MHZ="300.0"
    IMPL_STRATEGY="Performance_Explore"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_topdt_300_mem16)
    PROJECT_NAME="hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_topdt_300_mem16_overlay"
    ARTIFACT_NAME="hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_topdt_300_mem16"
    HLS_IP_REPO="generated/hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_topdt_300_mem16_no_board/solution_e2e_q4w8a/impl/ip"
    PL_CLK_MHZ="300.0"
    IMPL_STRATEGY="Performance_Explore"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_nodt_300_mem16)
    PROJECT_NAME="hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_nodt_300_mem16_overlay"
    ARTIFACT_NAME="hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_nodt_300_mem16"
    HLS_IP_REPO="generated/hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_nodt_300_mem16_no_board/solution_e2e_q4w8a/impl/ip"
    PL_CLK_MHZ="300.0"
    IMPL_STRATEGY="Performance_Explore"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normstage_300_mem16)
    PROJECT_NAME="hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normstage_300_mem16_overlay"
    ARTIFACT_NAME="hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normstage_300_mem16"
    HLS_IP_REPO="generated/hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normstage_300_mem16_no_board/solution_e2e_q4w8a/impl/ip"
    PL_CLK_MHZ="300.0"
    IMPL_STRATEGY="Performance_Explore"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_noobs_300_mem16)
    PROJECT_NAME="hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_noobs_300_mem16_overlay"
    ARTIFACT_NAME="hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_noobs_300_mem16"
    HLS_IP_REPO="generated/hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_noobs_300_mem16_no_board/solution_e2e_q4w8a/impl/ip"
    PL_CLK_MHZ="300.0"
    IMPL_STRATEGY="Performance_Explore"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_intnl_headraw_addrsw_300_mem16)
    PROJECT_NAME="hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_intnl_headraw_addrsw_300_mem16_overlay"
    ARTIFACT_NAME="hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_intnl_headraw_addrsw_300_mem16"
    HLS_IP_REPO="generated/hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_intnl_headraw_addrsw_300_mem16_no_board/solution_e2e_q4w8a/impl/ip"
    PL_CLK_MHZ="300.0"
    IMPL_STRATEGY="Performance_Explore"
    ;;
  par8_runtime_rom_dispatch_300_mem8)
    PROJECT_NAME="hgtxr_e2e_axis_dma_par8_runtime_rom_dispatch_300_mem8_overlay"
    ARTIFACT_NAME="hgtxr_e2e_axis_dma_par8_runtime_rom_dispatch_300_mem8"
    HLS_IP_REPO="generated/hgtxr_e2e_axis_par8_runtime_rom_dispatch_300_mem8_no_board/solution_e2e_q4w8a/impl/ip"
    PL_CLK_MHZ="300.0"
    ;;
  par8_runtime_rom_only_dispatch_300_mem8)
    PROJECT_NAME="hgtxr_e2e_axis_dma_par8_runtime_rom_only_dispatch_300_mem8_overlay"
    ARTIFACT_NAME="hgtxr_e2e_axis_dma_par8_runtime_rom_only_dispatch_300_mem8"
    HLS_IP_REPO="generated/hgtxr_e2e_axis_par8_runtime_rom_only_dispatch_300_mem8_no_board/solution_e2e_q4w8a/impl/ip"
    PL_CLK_MHZ="300.0"
    ;;
  par8_runtime_rom_only_dispatch_dsp3_300_mem8)
    PROJECT_NAME="hgtxr_e2e_axis_dma_par8_runtime_rom_only_dispatch_dsp3_300_mem8_overlay"
    ARTIFACT_NAME="hgtxr_e2e_axis_dma_par8_runtime_rom_only_dispatch_dsp3_300_mem8"
    HLS_IP_REPO="generated/hgtxr_e2e_axis_par8_runtime_rom_only_dispatch_dsp3_300_mem8_no_board/solution_e2e_q4w8a/impl/ip"
    PL_CLK_MHZ="300.0"
    IMPL_STRATEGY="Performance_Explore"
    ;;
  *)
    echo "unknown profile: $PROFILE" >&2
    echo "additional profile: par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail4_dsppipe4_300_mem16" >&2
    echo "additional profile: par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail8_lutbuf_dsppipe4_300_mem16" >&2
    echo "additional profile: par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail16_lutbuf_dsppipe4_300_mem16" >&2
    echo "additional profile: par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_toklutbuf_dsppipe4_300_mem16" >&2
    echo "additional profile: par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_dsppipe4_keep_300_mem16" >&2
    echo "additional profile: par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16" >&2
    echo "additional profile: par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_mtok4_dtok4_aq2_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only" >&2
    echo "additional profile: par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_mtok4_dtok4_aq2_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_track_only" >&2
    echo "profiles: par16_c3b_mem16 par32_dsp_mixed_stream_mem16 par32_runtime_mode_mem16 par32_runtime_full_axi_mem16 par32_runtime_full_axi_dsp_mem16 par32_runtime_rom_only_dispatch_dsp3_300_mem16 par32_runtime_rom_only_dispatch_dsp3_headlut_300_mem16 par32_runtime_rom_only_dispatch_dsp3_addrsw_300_mem16 par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_300_mem16 par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_300_mem16 par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_300_mem16 par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_300_mem16 par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_300_mem16 par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_300_mem16 par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_dsppipe4_300_mem16 par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_dsppipe4_300_mem16 par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_toklutbuf_dsppipe4_300_mem16 par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_300_mem16_search_only par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_pipepref_300_mem16 par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_pipepref_tail2_300_mem16 par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_pipepref_tail4_300_mem16 par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_pipepref_cttail4_300_mem16 par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_pipepref_tail8_300_mem16 par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_pipepref_coreallfabric_300_mem16 par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_pipepref_coreallfabric_nowide_300_mem16 par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_topdt_300_mem16 par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_nodt_300_mem16 par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normstage_300_mem16 par32_runtime_rom_only_dispatch_dsp3_intnl_headraw_addrsw_300_mem16 par8_runtime_rom_dispatch_300_mem8 par8_runtime_rom_only_dispatch_300_mem8 par8_runtime_rom_only_dispatch_dsp3_300_mem8" >&2
    exit 2
    ;;
esac

if [ -n "${HGTXR_VIVADO_PROJECT_NAME:-}" ]; then
  PROJECT_NAME="$HGTXR_VIVADO_PROJECT_NAME"
fi
if [ -n "${HGTXR_VIVADO_ARTIFACT_NAME:-}" ]; then
  ARTIFACT_NAME="$HGTXR_VIVADO_ARTIFACT_NAME"
fi
if [ -n "${HGTXR_HLS_IP_REPO:-}" ]; then
  HLS_IP_REPO="$HGTXR_HLS_IP_REPO"
fi

VIVADO_BIN=${VIVADO_BIN:-}
if [ -z "$VIVADO_BIN" ]; then
  if command -v vivado >/dev/null 2>&1; then
    VIVADO_BIN=$(command -v vivado)
  elif [ -x /tools/Xilinx/Vivado/2023.2/bin/vivado ]; then
    VIVADO_BIN=/tools/Xilinx/Vivado/2023.2/bin/vivado
  else
    echo "vivado not found. Set VIVADO_BIN or source Xilinx environment." >&2
    exit 127
  fi
fi

XILINX_ROOT=${HGTXR_XILINX_ROOT:-/tools/Xilinx}
VIVADO_LIB_DIR="$XILINX_ROOT/Vivado/2023.2/lib/lnx64.o/Rhel/9"
VITIS_HLS_LIB_DIR="$XILINX_ROOT/Vitis_HLS/2023.2/lib/lnx64.o/Rhel/9"
if [ -d "$VIVADO_LIB_DIR" ]; then
  export LD_LIBRARY_PATH="$VIVADO_LIB_DIR:${LD_LIBRARY_PATH:-}"
fi
if [ -d "$VITIS_HLS_LIB_DIR" ]; then
  export LD_LIBRARY_PATH="$VITIS_HLS_LIB_DIR:${LD_LIBRARY_PATH:-}"
fi

OUT_DIR=${HGTXR_VIVADO_OUT_DIR:-generated/build/vivado}
JOBS=${HGTXR_VIVADO_JOBS:-4}
PL_CLK_MHZ=${HGTXR_VIVADO_PL_CLK_MHZ:-${PL_CLK_MHZ:-100.0}}
IMPL_STRATEGY=${HGTXR_VIVADO_IMPL_STRATEGY:-${IMPL_STRATEGY:-default}}

echo "profile=$PROFILE"
echo "project_name=$PROJECT_NAME"
echo "artifact_name=$ARTIFACT_NAME"
echo "hls_ip_repo=$HLS_IP_REPO"
echo "out_dir=$OUT_DIR"
echo "jobs=$JOBS"
echo "pl_clk_mhz=$PL_CLK_MHZ"
echo "impl_strategy=$IMPL_STRATEGY"
echo "vivado_bin=$VIVADO_BIN"
echo "expected_bit=$OUT_DIR/overlay/$PROJECT_NAME/$ARTIFACT_NAME.bit"
echo "expected_hwh=$OUT_DIR/overlay/$PROJECT_NAME/$ARTIFACT_NAME.hwh"

if [ ! -f "$HLS_IP_REPO/component.xml" ]; then
  echo "missing packaged HLS IP: $HLS_IP_REPO/component.xml" >&2
  echo "run: scripts/run/run_e2e_q4w8a_no_board.sh package $PROFILE" >&2
  exit 3
fi

exec "$VIVADO_BIN" -mode batch -source vivado/scripts/build_e2e_axis_dma_bitstream.tcl -tclargs \
  -project_name "$PROJECT_NAME" \
  -artifact_name "$ARTIFACT_NAME" \
  -hls_ip_repo "$HLS_IP_REPO" \
  -out_dir "$OUT_DIR" \
  -jobs "$JOBS" \
  -pl_clk_mhz "$PL_CLK_MHZ" \
  -impl_strategy "$IMPL_STRATEGY"
