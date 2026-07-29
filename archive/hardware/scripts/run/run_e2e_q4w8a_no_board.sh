#!/bin/sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$ROOT_DIR"

MODE=${1:-csynth}
PROFILE=${2:-par32_dsp_uram_mem16}
EXTRA_CFLAGS=""

case "$MODE" in
  csim)
    HLS_TCL="vivado/scripts/run_e2e_q4w8a_csim.tcl"
    ;;
  csynth)
    HLS_TCL="vivado/scripts/run_e2e_q4w8a_csynth.tcl"
    ;;
  cosim)
    HLS_TCL="vivado/scripts/run_e2e_q4w8a_cosim.tcl"
    ;;
  package)
    HLS_TCL="vivado/scripts/package_e2e_axis_ip.tcl"
    ;;
  *)
    echo "usage: $0 {csim|csynth|cosim|package} [profile]" >&2
    echo "additional profile: par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail4_dsppipe4_300_mem16" >&2
    echo "additional profile: par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail8_lutbuf_dsppipe4_300_mem16" >&2
    echo "additional profile: par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail16_lutbuf_dsppipe4_300_mem16" >&2
    echo "additional profile: par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail16_lutbuf_shareunit_dsppipe4_300_mem16" >&2
    echo "additional profile: par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail16_softmaxq_lutbuf_dsppipe4_300_mem16" >&2
    echo "additional profile: par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail16_intnl_lutbuf_acc24_dsppipe4_300_mem16" >&2
    echo "additional profile: par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_coreallfabric_nowide_lutbuf_300_mem16" >&2
    echo "additional profile: par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_coreallfabric_intnl_lutbuf_acc24_300_mem16" >&2
    echo "additional profile: par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_toklutbuf_dsppipe4_300_mem16" >&2
    echo "additional profile: par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_toklutbuf_auxfabric_normlut_dsppipe4_300_mem16" >&2
    echo "additional profile: par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_toklutbuf_nocache_dsppipe4_300_mem16" >&2
    echo "additional profile: par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail8_lutbuf_auxfabric_normlut_dsppipe4_300_mem16" >&2
    echo "additional profile: par16_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_toklutbuf_intnl_acc24_dsppipe4_300_mem8" >&2
    echo "additional profile: par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_dsppipe4_keep_300_mem16" >&2
    echo "additional profile: par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_dsppipe4_300_mem16_search_only" >&2
    echo "additional profile: par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_dsppipe4_300_mem16_track_only" >&2
    echo "additional profile: par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar4_w2hgroup_mtok4_dtok4_aq4_qkvbram_tail2_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only" >&2
    echo "additional profile: par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar4_w2hgroup_mtok4_dtok4_aq4_qkvbram_tail2_patch64_attntok2_attnbram_nocache_patchpar64_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only" >&2
    echo "additional profile: par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar4_w2hgroup_mtok4_dtok4_aq4_qkvbram_tail2_exppart_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only" >&2
    echo "additional profile: par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar4_w2hgroup_mtok4_dtok4_aq4_qkvbram_tail2_exppart_skiph_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only" >&2
    echo "additional profile: par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar4_w2hgroup_mtok4_dtok4_aq4_qkvbram_tail2_exppart_w1c2_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only" >&2
    echo "additional profile: par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar4_w2hgroup_mtok4_dtok4_aq4_qkvbram_tail2_exppart_w1c3_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only" >&2
    echo "additional profile: par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar4_w2hgroup_mtok4_dtok4_aq6_qkvbram_tail2_exppart_w1c2_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only" >&2
    echo "additional profile: par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar4_w2hgroup_mtok4_dtok4_aq8_qkvbram_tail2_exppart_w1c2_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only" >&2
    echo "additional profile: par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar8_mlpfusebank_hpar8_w2hgroup_mtok4_dtok4_aq8_qkvbram_tail2_exppart_w1c2_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only" >&2
    echo "additional profile: par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar8_mlpfusebank_hpar8_acc24_w2hgroup_mtok4_dtok4_aq8_qkvbram_tail2_exppart_w1c2_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only" >&2
    echo "additional profile: par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar8_mlpfusebank_hpar16_acc24_w2hgroup_mtok4_dtok4_aq8_qkvbram_tail2_exppart_w1c2_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only" >&2
    echo "additional profile: par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar8_mlpfusebank_hpar16_acc24_gelupart_w2hgroup_mtok4_dtok4_aq8_qkvbram_tail2_exppart_w1c2_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only" >&2
    echo "additional profile: par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar8_mlpfusebank_hpar8_acc24_w2hgroup_mtok4_dtok4_aq8_qkvbram_tail2_exppart_w1c2_attntok2_attnbram_nocache_patchpar64_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only" >&2
    echo "additional profile: par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar8_mlpfusebank_hpar8_acc24_tokbank4_w2hgroup_mtok4_dtok4_aq8_qkvbram_tail2_exppart_w1c2_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only" >&2
    echo "additional profile: par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar8_mlpfusebank_hpar8_acc24_nowide_w2hgroup_mtok4_dtok4_aq8_qkvbram_tail2_exppart_w1c2_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only" >&2
    echo "additional profile: par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar8_mlpfusebank_hpar8_w2hgroup_mtok4_dtok4_aq8_qkvbram_tail2_exppart_w1c2_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_track_only" >&2
    echo "additional profile: par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar8_mlpfusebank_hpar8_skiph_w2hgroup_mtok4_dtok4_aq8_qkvbram_tail2_exppart_w1c2_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only" >&2
    echo "additional profile: par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar8_mlpfusebank_hpar8_lncache_w2hgroup_mtok4_dtok4_aq8_qkvbram_tail2_exppart_w1c2_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only" >&2
    echo "additional profile: par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar8_mlpfusebank_hpar8_w2fl2_w2hgroup_mtok4_dtok4_aq8_qkvbram_tail2_exppart_w1c2_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only" >&2
    echo "additional profile: par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar4_w2hgroup_mtok4_dtok4_aq8_qkvbram_tail8_exppart_w1c2_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only" >&2
    echo "additional profile: par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar4_w2hgroup_mtok4_dtok4_aq8_qkvbram_tail2_exppart_w1c4_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only" >&2
    echo "additional profile: par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar4_w2hgroup_mtok4_dtok4_aq8_qkvbram_tail2_exppart_w1c4_w1bank16_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only" >&2
    echo "additional profile: par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar4_w2hgroup_mtok4_dtok4_aq4_qkvbram_tail4_exppart_w1c2_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only" >&2
    echo "profiles: par16_c3b_recheck par32_runtime_mode_mem16 par32_runtime_full_axi_mem16 par32_runtime_full_axi_dsp_mem16 par32_runtime_rom_dispatch_300_mem16 par32_runtime_rom_only_dispatch_dsp3_300_mem16 par32_runtime_rom_only_dispatch_dsp3_headlut_300_mem16 par32_runtime_rom_only_dispatch_dsp3_addrsw_300_mem16 par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_300_mem16 par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_300_mem16 par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_300_mem16 par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_300_mem16 par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_structrom_300_mem16 par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_structrom_300_mem16_search_only par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_structrom_300_mem16_track_only par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_300_mem16 par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_300_mem16 par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_dsppipe4_300_mem16 par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_toklutbuf_dsppipe4_300_mem16 par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_300_mem16_search_only par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_300_mem16_search_only par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_300_mem16_track_only par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_pipepref_300_mem16 par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_pipepref_tail2_300_mem16 par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_pipepref_tail4_300_mem16 par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_pipepref_cttail4_300_mem16 par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_pipepref_tail8_300_mem16 par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_pipepref_coreallfabric_300_mem16 par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_pipepref_coreallfabric_nowide_300_mem16 par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_topdt_300_mem16 par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_nodt_300_mem16 par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normstage_300_mem16 par32_runtime_rom_only_dispatch_dsp3_intnl_headraw_addrsw_300_mem16 par32_runtime_rom_only_dispatch_dsp3_300_mem16_search_only par32_runtime_rom_only_dispatch_dsp3_300_mem16_track_only par16_runtime_rom_dispatch_300_mem16 par8_runtime_rom_dispatch_300_mem8 par8_runtime_rom_only_dispatch_300_mem8 par8_runtime_rom_only_dispatch_dsp3_300_mem8 par8_runtime_rom_only_dispatch_dsp3_300_mem8_search_only par8_runtime_rom_only_dispatch_dsp3_300_mem8_track_only par32_dsp_uram_mem16 par32_dsp_uram_mem32 par32_dsp_mixed_stream_mem16 par32_dsp_mixed_mem16 par32_dsp_mixed_mem32" >&2
    exit 2
    ;;
esac

case "$PROFILE" in
  par16_c3b_recheck)
    PROJECT="hgtxr_e2e_axis_par16_c3b_recheck"
    E2E_PAR=16
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_uram"
    E2E_SCALE="full"
    ;;
  par32_runtime_mode_mem16)
    PROJECT="hgtxr_e2e_axis_par32_runtime_mode_mem16_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_stream"
    E2E_SCALE="runtime_mode_active64_b8_ff768"
    ;;
  par32_runtime_full_axi_mem16)
    PROJECT="hgtxr_e2e_axis_par32_runtime_full_axi_mem16_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_stream"
    E2E_SCALE="runtime_full_axi_active64_b8_ff768"
    ;;
  par32_runtime_full_axi_dsp_mem16)
    PROJECT="hgtxr_e2e_axis_par32_runtime_full_axi_dsp_mem16_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_stream"
    E2E_SCALE="runtime_full_axi_dsp_active64_b8_ff768"
    HLS_CLOCK_PERIOD=5.0
    ;;
  par32_runtime_rom_dispatch_300_mem16)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_dispatch_300_mem16_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_stream"
    E2E_SCALE="runtime_rom_dispatch_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    ;;
  par32_runtime_rom_only_dispatch_dsp3_300_mem16)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_300_mem16_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_stream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=datapath
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headlut_300_mem16)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headlut_300_mem16_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_stream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headlut_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=datapath
    ;;
  par32_runtime_rom_only_dispatch_dsp3_addrsw_300_mem16)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_addrsw_300_mem16_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_stream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_addrsw_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=datapath
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_300_mem16)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_300_mem16_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_stream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=datapath
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_300_mem16)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_300_mem16_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_stream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=datapath
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_300_mem16)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_300_mem16_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=datapath
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_300_mem16)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_300_mem16_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_structrom_300_mem16)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_structrom_300_mem16_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-DHGTXR_E2E_STRUCTURED_PARAM_ROM=1 -DHGTXR_E2E_STRUCTURED_FAST_MATH=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_structrom_300_mem16_search_only)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_structrom_300_mem16_search_only_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-DHGTXR_E2E_STRUCTURED_PARAM_ROM=1 -DHGTXR_E2E_STRUCTURED_FAST_MATH=1 -UHGTXR_E2E_TEST_RUNTIME_MODES -DHGTXR_E2E_TEST_RUNTIME_MODES=0 -DHGTXR_E2E_FORCE_RUNTIME_MODE=0 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_structrom_300_mem16_track_only)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_structrom_300_mem16_track_only_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-DHGTXR_E2E_STRUCTURED_PARAM_ROM=1 -DHGTXR_E2E_STRUCTURED_FAST_MATH=1 -UHGTXR_E2E_TEST_RUNTIME_MODES -DHGTXR_E2E_TEST_RUNTIME_MODES=0 -DHGTXR_E2E_FORCE_RUNTIME_MODE=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_300_mem16)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_300_mem16_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_300_mem16)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_300_mem16_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_CORE_FABRIC_TAIL_LANES=2 -DHGTXR_E2E_CORE_LANE_CT_SWITCH=1 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_dsppipe4_300_mem16)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_dsppipe4_300_mem16_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_CORE_FABRIC_TAIL_LANES=2 -DHGTXR_E2E_CORE_LANE_CT_SWITCH=1 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_toklutbuf_dsppipe4_300_mem16)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_toklutbuf_dsppipe4_300_mem16_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_CORE_FABRIC_TAIL_LANES=2 -DHGTXR_E2E_CORE_LANE_CT_SWITCH=1 -DHGTXR_E2E_LUTRAM_FRAME_TOKENS=1 -DHGTXR_E2E_LUTRAM_GB_TOKENS=1 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_toklutbuf_auxfabric_normlut_dsppipe4_300_mem16)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_toklutbuf_auxfabric_normlut_dsppipe4_300_mem16_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_PATCH_FABRIC_MUL=1 -DHGTXR_E2E_LAYERNORM_FABRIC_MUL=1 -DHGTXR_E2E_FASTPATH_FABRIC_MUL=1 -DHGTXR_E2E_CORE_FABRIC_TAIL_LANES=2 -DHGTXR_E2E_CORE_LANE_CT_SWITCH=1 -DHGTXR_E2E_LUTRAM_FRAME_TOKENS=1 -DHGTXR_E2E_LUTRAM_GB_TOKENS=1 -DHGTXR_E2E_LUTRAM_NORM=1 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_toklutbuf_nocache_dsppipe4_300_mem16)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_toklutbuf_nocache_dsppipe4_300_mem16_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_WEIGHT_VEC_CACHE -DHGTXR_E2E_WEIGHT_VEC_CACHE=0 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_CORE_FABRIC_TAIL_LANES=2 -DHGTXR_E2E_CORE_LANE_CT_SWITCH=1 -DHGTXR_E2E_LUTRAM_FRAME_TOKENS=1 -DHGTXR_E2E_LUTRAM_GB_TOKENS=1 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail4_dsppipe4_300_mem16)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail4_dsppipe4_300_mem16_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_CORE_FABRIC_TAIL_LANES=4 -DHGTXR_E2E_CORE_LANE_CT_SWITCH=1 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail8_lutbuf_dsppipe4_300_mem16)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail8_lutbuf_dsppipe4_300_mem16_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_CORE_FABRIC_TAIL_LANES=8 -DHGTXR_E2E_CORE_LANE_CT_SWITCH=1 -DHGTXR_E2E_LUTRAM_FRAME_TOKENS=1 -DHGTXR_E2E_LUTRAM_GB_TOKENS=1 -DHGTXR_E2E_LUTRAM_NORM=1 -DHGTXR_E2E_LUTRAM_Q=1 -DHGTXR_E2E_LUTRAM_K=1 -DHGTXR_E2E_LUTRAM_V=1 -DHGTXR_E2E_LUTRAM_ATTN=1 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail8_lutbuf_auxfabric_normlut_dsppipe4_300_mem16)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail8_lutbuf_auxfabric_normlut_dsppipe4_300_mem16_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_PATCH_FABRIC_MUL=1 -DHGTXR_E2E_LAYERNORM_FABRIC_MUL=1 -DHGTXR_E2E_FASTPATH_FABRIC_MUL=1 -DHGTXR_E2E_CORE_FABRIC_TAIL_LANES=8 -DHGTXR_E2E_CORE_LANE_CT_SWITCH=1 -DHGTXR_E2E_LUTRAM_FRAME_TOKENS=1 -DHGTXR_E2E_LUTRAM_GB_TOKENS=1 -DHGTXR_E2E_LUTRAM_NORM=1 -DHGTXR_E2E_LUTRAM_Q=1 -DHGTXR_E2E_LUTRAM_K=1 -DHGTXR_E2E_LUTRAM_V=1 -DHGTXR_E2E_LUTRAM_ATTN=1 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail16_lutbuf_dsppipe4_300_mem16)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail16_lutbuf_dsppipe4_300_mem16_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_CORE_FABRIC_TAIL_LANES=16 -DHGTXR_E2E_CORE_LANE_CT_SWITCH=1 -DHGTXR_E2E_LUTRAM_FRAME_TOKENS=1 -DHGTXR_E2E_LUTRAM_GB_TOKENS=1 -DHGTXR_E2E_LUTRAM_NORM=1 -DHGTXR_E2E_LUTRAM_Q=1 -DHGTXR_E2E_LUTRAM_K=1 -DHGTXR_E2E_LUTRAM_V=1 -DHGTXR_E2E_LUTRAM_ATTN=1 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail16_lutbuf_shareunit_dsppipe4_300_mem16)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail16_lutbuf_shareunit_dsppipe4_300_mem16_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_CORE_FABRIC_TAIL_LANES=16 -DHGTXR_E2E_CORE_LANE_CT_SWITCH=1 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -DHGTXR_E2E_LUTRAM_FRAME_TOKENS=1 -DHGTXR_E2E_LUTRAM_GB_TOKENS=1 -DHGTXR_E2E_LUTRAM_NORM=1 -DHGTXR_E2E_LUTRAM_Q=1 -DHGTXR_E2E_LUTRAM_K=1 -DHGTXR_E2E_LUTRAM_V=1 -DHGTXR_E2E_LUTRAM_ATTN=1 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_lutbuf_shareunit_dsppipe4_300_mem16)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_lutbuf_shareunit_dsppipe4_300_mem16_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -DHGTXR_E2E_LUTRAM_FRAME_TOKENS=1 -DHGTXR_E2E_LUTRAM_GB_TOKENS=1 -DHGTXR_E2E_LUTRAM_NORM=1 -DHGTXR_E2E_LUTRAM_Q=1 -DHGTXR_E2E_LUTRAM_K=1 -DHGTXR_E2E_LUTRAM_V=1 -DHGTXR_E2E_LUTRAM_ATTN=1 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_lutbuf_shareunit_headpar8_dsppipe4_300_mem16)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_lutbuf_shareunit_headpar8_dsppipe4_300_mem16_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -DHGTXR_E2E_HEAD_PAR=8 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -DHGTXR_E2E_LUTRAM_FRAME_TOKENS=1 -DHGTXR_E2E_LUTRAM_GB_TOKENS=1 -DHGTXR_E2E_LUTRAM_NORM=1 -DHGTXR_E2E_LUTRAM_Q=1 -DHGTXR_E2E_LUTRAM_K=1 -DHGTXR_E2E_LUTRAM_V=1 -DHGTXR_E2E_LUTRAM_ATTN=1 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_dsppipe4_300_mem16)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_dsppipe4_300_mem16_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -DHGTXR_E2E_HEAD_PAR=8 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -DHGTXR_E2E_LUTRAM_NORM=1 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_dsppipe4_300_mem16)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_dsppipe4_300_mem16_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -DHGTXR_E2E_HEAD_PAR=8 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -DHGTXR_E2E_LUTRAM_NORM=1 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_dsppipe4_300_mem16_search_only)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_dsppipe4_300_mem16_search_only_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -DHGTXR_E2E_HEAD_PAR=8 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -DHGTXR_E2E_LUTRAM_NORM=1 -UHGTXR_E2E_TEST_RUNTIME_MODES -DHGTXR_E2E_TEST_RUNTIME_MODES=0 -DHGTXR_E2E_FORCE_RUNTIME_MODE=0 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_dsppipe4_300_mem16_track_only)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_dsppipe4_300_mem16_track_only_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -DHGTXR_E2E_HEAD_PAR=8 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -DHGTXR_E2E_LUTRAM_NORM=1 -UHGTXR_E2E_TEST_RUNTIME_MODES -DHGTXR_E2E_TEST_RUNTIME_MODES=0 -DHGTXR_E2E_FORCE_RUNTIME_MODE=1 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_patchpar8_dsppipe4_300_mem16)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_patchpar8_dsppipe4_300_mem16_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -DHGTXR_E2E_HEAD_PAR=8 -DHGTXR_E2E_PATCH_PAR=8 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -DHGTXR_E2E_LUTRAM_NORM=1 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_patchpar8_dsppipe4_300_mem16_search_only)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_patchpar8_dsppipe4_300_mem16_search_only_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -DHGTXR_E2E_HEAD_PAR=8 -DHGTXR_E2E_PATCH_PAR=8 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -DHGTXR_E2E_LUTRAM_NORM=1 -UHGTXR_E2E_TEST_RUNTIME_MODES -DHGTXR_E2E_TEST_RUNTIME_MODES=0 -DHGTXR_E2E_FORCE_RUNTIME_MODE=0 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_patchpar8_dsppipe4_300_mem16_track_only)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_patchpar8_dsppipe4_300_mem16_track_only_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -DHGTXR_E2E_HEAD_PAR=8 -DHGTXR_E2E_PATCH_PAR=8 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -DHGTXR_E2E_LUTRAM_NORM=1 -UHGTXR_E2E_TEST_RUNTIME_MODES -DHGTXR_E2E_TEST_RUNTIME_MODES=0 -DHGTXR_E2E_FORCE_RUNTIME_MODE=1 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_patchpar8_tokenloop_dsppipe4_300_mem16)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_patchpar8_tokenloop_dsppipe4_300_mem16_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -DHGTXR_E2E_HEAD_PAR=8 -DHGTXR_E2E_PATCH_PAR=8 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -DHGTXR_E2E_LUTRAM_NORM=1 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_patchpar8_tokenloop_dsppipe4_300_mem16_search_only)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_patchpar8_tokenloop_dsppipe4_300_mem16_search_only_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -DHGTXR_E2E_HEAD_PAR=8 -DHGTXR_E2E_PATCH_PAR=8 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -DHGTXR_E2E_LUTRAM_NORM=1 -UHGTXR_E2E_TEST_RUNTIME_MODES -DHGTXR_E2E_TEST_RUNTIME_MODES=0 -DHGTXR_E2E_FORCE_RUNTIME_MODE=0 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_patchpar8_tokenloop_dsppipe4_300_mem16_track_only)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_patchpar8_tokenloop_dsppipe4_300_mem16_track_only_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -DHGTXR_E2E_HEAD_PAR=8 -DHGTXR_E2E_PATCH_PAR=8 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -DHGTXR_E2E_LUTRAM_NORM=1 -UHGTXR_E2E_TEST_RUNTIME_MODES -DHGTXR_E2E_TEST_RUNTIME_MODES=0 -DHGTXR_E2E_FORCE_RUNTIME_MODE=1 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_patchpar8_tokenloop_dense64_dsppipe4_300_mem16)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_patchpar8_tokenloop_dense64_dsppipe4_300_mem16_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_DENSE_PAR -DHGTXR_E2E_DENSE_PAR=64 -UHGTXR_E2E_ATTN_PAR -DHGTXR_E2E_ATTN_PAR=64 -DHGTXR_E2E_HEAD_PAR=8 -DHGTXR_E2E_PATCH_PAR=8 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -DHGTXR_E2E_LUTRAM_NORM=1 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_patchpar8_tokenloop_dense64_dsppipe4_300_mem16_search_only)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_patchpar8_tokenloop_dense64_dsppipe4_300_mem16_search_only_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_DENSE_PAR -DHGTXR_E2E_DENSE_PAR=64 -UHGTXR_E2E_ATTN_PAR -DHGTXR_E2E_ATTN_PAR=64 -DHGTXR_E2E_HEAD_PAR=8 -DHGTXR_E2E_PATCH_PAR=8 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -DHGTXR_E2E_LUTRAM_NORM=1 -UHGTXR_E2E_TEST_RUNTIME_MODES -DHGTXR_E2E_TEST_RUNTIME_MODES=0 -DHGTXR_E2E_FORCE_RUNTIME_MODE=0 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_patchpar8_tokenloop_dense64_dsppipe4_300_mem16_track_only)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_patchpar8_tokenloop_dense64_dsppipe4_300_mem16_track_only_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_DENSE_PAR -DHGTXR_E2E_DENSE_PAR=64 -UHGTXR_E2E_ATTN_PAR -DHGTXR_E2E_ATTN_PAR=64 -DHGTXR_E2E_HEAD_PAR=8 -DHGTXR_E2E_PATCH_PAR=8 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -DHGTXR_E2E_LUTRAM_NORM=1 -UHGTXR_E2E_TEST_RUNTIME_MODES -DHGTXR_E2E_TEST_RUNTIME_MODES=0 -DHGTXR_E2E_FORCE_RUNTIME_MODE=1 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_patchpar16_tokenloop_dsppipe4_300_mem16)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_patchpar16_tokenloop_dsppipe4_300_mem16_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -DHGTXR_E2E_HEAD_PAR=8 -DHGTXR_E2E_PATCH_PAR=16 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -DHGTXR_E2E_LUTRAM_NORM=1 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_patchpar16_tokenloop_dsppipe4_300_mem16_search_only)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_patchpar16_tokenloop_dsppipe4_300_mem16_search_only_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -DHGTXR_E2E_HEAD_PAR=8 -DHGTXR_E2E_PATCH_PAR=16 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -DHGTXR_E2E_LUTRAM_NORM=1 -UHGTXR_E2E_TEST_RUNTIME_MODES -DHGTXR_E2E_TEST_RUNTIME_MODES=0 -DHGTXR_E2E_FORCE_RUNTIME_MODE=0 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_patchpar16_tokenloop_dsppipe4_300_mem16_track_only)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_patchpar16_tokenloop_dsppipe4_300_mem16_track_only_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -DHGTXR_E2E_HEAD_PAR=8 -DHGTXR_E2E_PATCH_PAR=16 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -DHGTXR_E2E_LUTRAM_NORM=1 -UHGTXR_E2E_TEST_RUNTIME_MODES -DHGTXR_E2E_TEST_RUNTIME_MODES=0 -DHGTXR_E2E_FORCE_RUNTIME_MODE=1 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=4 -DHGTXR_E2E_PATCH_PAR=16 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=4 -DHGTXR_E2E_PATCH_PAR=16 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -UHGTXR_E2E_TEST_RUNTIME_MODES -DHGTXR_E2E_TEST_RUNTIME_MODES=0 -DHGTXR_E2E_FORCE_RUNTIME_MODE=0 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_track_only)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_track_only_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=4 -DHGTXR_E2E_PATCH_PAR=16 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -UHGTXR_E2E_TEST_RUNTIME_MODES -DHGTXR_E2E_TEST_RUNTIME_MODES=0 -DHGTXR_E2E_FORCE_RUNTIME_MODE=1 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlptok2_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlptok2_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=4 -DHGTXR_E2E_PATCH_PAR=16 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -DHGTXR_E2E_MLP_TOKEN_PAR=2 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlptok2_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlptok2_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=4 -DHGTXR_E2E_PATCH_PAR=16 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -DHGTXR_E2E_MLP_TOKEN_PAR=2 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -UHGTXR_E2E_TEST_RUNTIME_MODES -DHGTXR_E2E_TEST_RUNTIME_MODES=0 -DHGTXR_E2E_FORCE_RUNTIME_MODE=0 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlptok2_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_track_only)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlptok2_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_track_only_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=4 -DHGTXR_E2E_PATCH_PAR=16 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -DHGTXR_E2E_MLP_TOKEN_PAR=2 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -UHGTXR_E2E_TEST_RUNTIME_MODES -DHGTXR_E2E_TEST_RUNTIME_MODES=0 -DHGTXR_E2E_FORCE_RUNTIME_MODE=1 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfuse_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfuse_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=4 -DHGTXR_E2E_PATCH_PAR=16 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -DHGTXR_E2E_MLP_FUSED_W2=1 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfuse_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfuse_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=4 -DHGTXR_E2E_PATCH_PAR=16 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -DHGTXR_E2E_MLP_FUSED_W2=1 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -UHGTXR_E2E_TEST_RUNTIME_MODES -DHGTXR_E2E_TEST_RUNTIME_MODES=0 -DHGTXR_E2E_FORCE_RUNTIME_MODE=0 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfuse_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_track_only)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfuse_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_track_only_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=4 -DHGTXR_E2E_PATCH_PAR=16 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -DHGTXR_E2E_MLP_FUSED_W2=1 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -UHGTXR_E2E_TEST_RUNTIME_MODES -DHGTXR_E2E_TEST_RUNTIME_MODES=0 -DHGTXR_E2E_FORCE_RUNTIME_MODE=1 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=4 -DHGTXR_E2E_PATCH_PAR=16 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -DHGTXR_E2E_MLP_FUSED_W2=1 -DHGTXR_E2E_MLP_FUSED_W2_BANKED_ACC=1 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=4 -DHGTXR_E2E_PATCH_PAR=16 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -DHGTXR_E2E_MLP_FUSED_W2=1 -DHGTXR_E2E_MLP_FUSED_W2_BANKED_ACC=1 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -UHGTXR_E2E_TEST_RUNTIME_MODES -DHGTXR_E2E_TEST_RUNTIME_MODES=0 -DHGTXR_E2E_FORCE_RUNTIME_MODE=0 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_track_only)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_track_only_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=4 -DHGTXR_E2E_PATCH_PAR=16 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -DHGTXR_E2E_MLP_FUSED_W2=1 -DHGTXR_E2E_MLP_FUSED_W2_BANKED_ACC=1 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -UHGTXR_E2E_TEST_RUNTIME_MODES -DHGTXR_E2E_TEST_RUNTIME_MODES=0 -DHGTXR_E2E_FORCE_RUNTIME_MODE=1 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=4 -DHGTXR_E2E_PATCH_PAR=16 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -DHGTXR_E2E_MLP_FUSED_W2=1 -DHGTXR_E2E_MLP_FUSED_W2_BANKED_ACC=1 -DHGTXR_E2E_MLP_FUSED_W2_HP_PAR=2 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=4 -DHGTXR_E2E_PATCH_PAR=16 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -DHGTXR_E2E_MLP_FUSED_W2=1 -DHGTXR_E2E_MLP_FUSED_W2_BANKED_ACC=1 -DHGTXR_E2E_MLP_FUSED_W2_HP_PAR=2 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -UHGTXR_E2E_TEST_RUNTIME_MODES -DHGTXR_E2E_TEST_RUNTIME_MODES=0 -DHGTXR_E2E_FORCE_RUNTIME_MODE=0 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_track_only)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_track_only_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=4 -DHGTXR_E2E_PATCH_PAR=16 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -DHGTXR_E2E_MLP_FUSED_W2=1 -DHGTXR_E2E_MLP_FUSED_W2_BANKED_ACC=1 -DHGTXR_E2E_MLP_FUSED_W2_HP_PAR=2 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -UHGTXR_E2E_TEST_RUNTIME_MODES -DHGTXR_E2E_TEST_RUNTIME_MODES=0 -DHGTXR_E2E_FORCE_RUNTIME_MODE=1 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_tok2_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_tok2_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=4 -DHGTXR_E2E_PATCH_PAR=16 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -DHGTXR_E2E_MLP_TOKEN_PAR=2 -DHGTXR_E2E_MLP_FUSED_W2=1 -DHGTXR_E2E_MLP_FUSED_W2_BANKED_ACC=1 -DHGTXR_E2E_MLP_FUSED_W2_HP_PAR=2 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_tok2_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_tok2_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=4 -DHGTXR_E2E_PATCH_PAR=16 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -DHGTXR_E2E_MLP_TOKEN_PAR=2 -DHGTXR_E2E_MLP_FUSED_W2=1 -DHGTXR_E2E_MLP_FUSED_W2_BANKED_ACC=1 -DHGTXR_E2E_MLP_FUSED_W2_HP_PAR=2 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -UHGTXR_E2E_TEST_RUNTIME_MODES -DHGTXR_E2E_TEST_RUNTIME_MODES=0 -DHGTXR_E2E_FORCE_RUNTIME_MODE=0 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_tok2_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_track_only)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_tok2_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_track_only_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=4 -DHGTXR_E2E_PATCH_PAR=16 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -DHGTXR_E2E_MLP_TOKEN_PAR=2 -DHGTXR_E2E_MLP_FUSED_W2=1 -DHGTXR_E2E_MLP_FUSED_W2_BANKED_ACC=1 -DHGTXR_E2E_MLP_FUSED_W2_HP_PAR=2 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -UHGTXR_E2E_TEST_RUNTIME_MODES -DHGTXR_E2E_TEST_RUNTIME_MODES=0 -DHGTXR_E2E_FORCE_RUNTIME_MODE=1 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_tok2_attntok2_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_tok2_attntok2_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=4 -DHGTXR_E2E_PATCH_PAR=16 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -DHGTXR_E2E_MLP_TOKEN_PAR=2 -DHGTXR_E2E_DENSE_TOKEN_PAR=2 -DHGTXR_E2E_MLP_FUSED_W2=1 -DHGTXR_E2E_MLP_FUSED_W2_BANKED_ACC=1 -DHGTXR_E2E_MLP_FUSED_W2_HP_PAR=2 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_tok2_attntok2_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_tok2_attntok2_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=4 -DHGTXR_E2E_PATCH_PAR=16 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -DHGTXR_E2E_MLP_TOKEN_PAR=2 -DHGTXR_E2E_DENSE_TOKEN_PAR=2 -DHGTXR_E2E_MLP_FUSED_W2=1 -DHGTXR_E2E_MLP_FUSED_W2_BANKED_ACC=1 -DHGTXR_E2E_MLP_FUSED_W2_HP_PAR=2 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -UHGTXR_E2E_TEST_RUNTIME_MODES -DHGTXR_E2E_TEST_RUNTIME_MODES=0 -DHGTXR_E2E_FORCE_RUNTIME_MODE=0 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_tok2_attntok2_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_track_only)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_tok2_attntok2_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_track_only_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=4 -DHGTXR_E2E_PATCH_PAR=16 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -DHGTXR_E2E_MLP_TOKEN_PAR=2 -DHGTXR_E2E_DENSE_TOKEN_PAR=2 -DHGTXR_E2E_MLP_FUSED_W2=1 -DHGTXR_E2E_MLP_FUSED_W2_BANKED_ACC=1 -DHGTXR_E2E_MLP_FUSED_W2_HP_PAR=2 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -UHGTXR_E2E_TEST_RUNTIME_MODES -DHGTXR_E2E_TEST_RUNTIME_MODES=0 -DHGTXR_E2E_FORCE_RUNTIME_MODE=1 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_tok2_attntok2_attnbram_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16)
    PROJECT="hgtxr_e2e_axis_par32_rt_attntok2_attnbram_300_mem16_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_URAM_ATTN -DHGTXR_E2E_URAM_ATTN=0 -UHGTXR_E2E_LUTRAM_ATTN -DHGTXR_E2E_LUTRAM_ATTN=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=4 -DHGTXR_E2E_PATCH_PAR=16 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -DHGTXR_E2E_MLP_TOKEN_PAR=2 -DHGTXR_E2E_DENSE_TOKEN_PAR=2 -DHGTXR_E2E_MLP_FUSED_W2=1 -DHGTXR_E2E_MLP_FUSED_W2_BANKED_ACC=1 -DHGTXR_E2E_MLP_FUSED_W2_HP_PAR=2 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_tok2_attntok2_attnbram_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only)
    PROJECT="hgtxr_e2e_axis_par32_rt_attntok2_attnbram_300_mem16_search_only_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_URAM_ATTN -DHGTXR_E2E_URAM_ATTN=0 -UHGTXR_E2E_LUTRAM_ATTN -DHGTXR_E2E_LUTRAM_ATTN=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=4 -DHGTXR_E2E_PATCH_PAR=16 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -DHGTXR_E2E_MLP_TOKEN_PAR=2 -DHGTXR_E2E_DENSE_TOKEN_PAR=2 -DHGTXR_E2E_MLP_FUSED_W2=1 -DHGTXR_E2E_MLP_FUSED_W2_BANKED_ACC=1 -DHGTXR_E2E_MLP_FUSED_W2_HP_PAR=2 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -UHGTXR_E2E_TEST_RUNTIME_MODES -DHGTXR_E2E_TEST_RUNTIME_MODES=0 -DHGTXR_E2E_FORCE_RUNTIME_MODE=0 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_tok2_attntok2_attnbram_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_track_only)
    PROJECT="hgtxr_e2e_axis_par32_rt_attntok2_attnbram_300_mem16_track_only_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_URAM_ATTN -DHGTXR_E2E_URAM_ATTN=0 -UHGTXR_E2E_LUTRAM_ATTN -DHGTXR_E2E_LUTRAM_ATTN=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=4 -DHGTXR_E2E_PATCH_PAR=16 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -DHGTXR_E2E_MLP_TOKEN_PAR=2 -DHGTXR_E2E_DENSE_TOKEN_PAR=2 -DHGTXR_E2E_MLP_FUSED_W2=1 -DHGTXR_E2E_MLP_FUSED_W2_BANKED_ACC=1 -DHGTXR_E2E_MLP_FUSED_W2_HP_PAR=2 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -UHGTXR_E2E_TEST_RUNTIME_MODES -DHGTXR_E2E_TEST_RUNTIME_MODES=0 -DHGTXR_E2E_FORCE_RUNTIME_MODE=1 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_tok2_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16)
    PROJECT="hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_300_mem16_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_URAM_ATTN -DHGTXR_E2E_URAM_ATTN=0 -UHGTXR_E2E_LUTRAM_ATTN -DHGTXR_E2E_LUTRAM_ATTN=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=4 -UHGTXR_E2E_PATCH_PAR -DHGTXR_E2E_PATCH_PAR=32 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -DHGTXR_E2E_MLP_TOKEN_PAR=2 -DHGTXR_E2E_DENSE_TOKEN_PAR=2 -DHGTXR_E2E_MLP_FUSED_W2=1 -DHGTXR_E2E_MLP_FUSED_W2_BANKED_ACC=1 -DHGTXR_E2E_MLP_FUSED_W2_HP_PAR=2 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_tok2_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only)
    PROJECT="hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_300_mem16_search_only_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_URAM_ATTN -DHGTXR_E2E_URAM_ATTN=0 -UHGTXR_E2E_LUTRAM_ATTN -DHGTXR_E2E_LUTRAM_ATTN=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=4 -UHGTXR_E2E_PATCH_PAR -DHGTXR_E2E_PATCH_PAR=32 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -DHGTXR_E2E_MLP_TOKEN_PAR=2 -DHGTXR_E2E_DENSE_TOKEN_PAR=2 -DHGTXR_E2E_MLP_FUSED_W2=1 -DHGTXR_E2E_MLP_FUSED_W2_BANKED_ACC=1 -DHGTXR_E2E_MLP_FUSED_W2_HP_PAR=2 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -UHGTXR_E2E_TEST_RUNTIME_MODES -DHGTXR_E2E_TEST_RUNTIME_MODES=0 -DHGTXR_E2E_FORCE_RUNTIME_MODE=0 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_tok2_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_track_only)
    PROJECT="hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_300_mem16_track_only_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_URAM_ATTN -DHGTXR_E2E_URAM_ATTN=0 -UHGTXR_E2E_LUTRAM_ATTN -DHGTXR_E2E_LUTRAM_ATTN=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=4 -UHGTXR_E2E_PATCH_PAR -DHGTXR_E2E_PATCH_PAR=32 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -DHGTXR_E2E_MLP_TOKEN_PAR=2 -DHGTXR_E2E_DENSE_TOKEN_PAR=2 -DHGTXR_E2E_MLP_FUSED_W2=1 -DHGTXR_E2E_MLP_FUSED_W2_BANKED_ACC=1 -DHGTXR_E2E_MLP_FUSED_W2_HP_PAR=2 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -UHGTXR_E2E_TEST_RUNTIME_MODES -DHGTXR_E2E_TEST_RUNTIME_MODES=0 -DHGTXR_E2E_FORCE_RUNTIME_MODE=1 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_mtok2_dtok4_attntok2_attnbram_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16)
    PROJECT="hgtxr_e2e_axis_par32_rt_attntok2_attnbram_dtok4_300_mem16_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_URAM_ATTN -DHGTXR_E2E_URAM_ATTN=0 -UHGTXR_E2E_LUTRAM_ATTN -DHGTXR_E2E_LUTRAM_ATTN=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=4 -DHGTXR_E2E_PATCH_PAR=16 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -UHGTXR_E2E_MLP_TOKEN_PAR -DHGTXR_E2E_MLP_TOKEN_PAR=2 -UHGTXR_E2E_DENSE_TOKEN_PAR -DHGTXR_E2E_DENSE_TOKEN_PAR=4 -DHGTXR_E2E_MLP_FUSED_W2=1 -DHGTXR_E2E_MLP_FUSED_W2_BANKED_ACC=1 -DHGTXR_E2E_MLP_FUSED_W2_HP_PAR=2 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_mtok2_dtok4_attntok2_attnbram_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only)
    PROJECT="hgtxr_e2e_axis_par32_rt_attntok2_attnbram_dtok4_300_mem16_search_only_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_URAM_ATTN -DHGTXR_E2E_URAM_ATTN=0 -UHGTXR_E2E_LUTRAM_ATTN -DHGTXR_E2E_LUTRAM_ATTN=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=4 -DHGTXR_E2E_PATCH_PAR=16 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -UHGTXR_E2E_MLP_TOKEN_PAR -DHGTXR_E2E_MLP_TOKEN_PAR=2 -UHGTXR_E2E_DENSE_TOKEN_PAR -DHGTXR_E2E_DENSE_TOKEN_PAR=4 -DHGTXR_E2E_MLP_FUSED_W2=1 -DHGTXR_E2E_MLP_FUSED_W2_BANKED_ACC=1 -DHGTXR_E2E_MLP_FUSED_W2_HP_PAR=2 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -UHGTXR_E2E_TEST_RUNTIME_MODES -DHGTXR_E2E_TEST_RUNTIME_MODES=0 -DHGTXR_E2E_FORCE_RUNTIME_MODE=0 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_mtok2_dtok4_attntok2_attnbram_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_track_only)
    PROJECT="hgtxr_e2e_axis_par32_rt_attntok2_attnbram_dtok4_300_mem16_track_only_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_URAM_ATTN -DHGTXR_E2E_URAM_ATTN=0 -UHGTXR_E2E_LUTRAM_ATTN -DHGTXR_E2E_LUTRAM_ATTN=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=4 -DHGTXR_E2E_PATCH_PAR=16 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -UHGTXR_E2E_MLP_TOKEN_PAR -DHGTXR_E2E_MLP_TOKEN_PAR=2 -UHGTXR_E2E_DENSE_TOKEN_PAR -DHGTXR_E2E_DENSE_TOKEN_PAR=4 -DHGTXR_E2E_MLP_FUSED_W2=1 -DHGTXR_E2E_MLP_FUSED_W2_BANKED_ACC=1 -DHGTXR_E2E_MLP_FUSED_W2_HP_PAR=2 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -UHGTXR_E2E_TEST_RUNTIME_MODES -DHGTXR_E2E_TEST_RUNTIME_MODES=0 -DHGTXR_E2E_FORCE_RUNTIME_MODE=1 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_mtok2_dtok4_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16)
    PROJECT="hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_dtok4_300_mem16_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_URAM_ATTN -DHGTXR_E2E_URAM_ATTN=0 -UHGTXR_E2E_LUTRAM_ATTN -DHGTXR_E2E_LUTRAM_ATTN=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=4 -UHGTXR_E2E_PATCH_PAR -DHGTXR_E2E_PATCH_PAR=32 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -UHGTXR_E2E_MLP_TOKEN_PAR -DHGTXR_E2E_MLP_TOKEN_PAR=2 -UHGTXR_E2E_DENSE_TOKEN_PAR -DHGTXR_E2E_DENSE_TOKEN_PAR=4 -DHGTXR_E2E_MLP_FUSED_W2=1 -DHGTXR_E2E_MLP_FUSED_W2_BANKED_ACC=1 -DHGTXR_E2E_MLP_FUSED_W2_HP_PAR=2 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_mtok2_dtok4_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only)
    PROJECT="hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_dtok4_300_mem16_search_only_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_URAM_ATTN -DHGTXR_E2E_URAM_ATTN=0 -UHGTXR_E2E_LUTRAM_ATTN -DHGTXR_E2E_LUTRAM_ATTN=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=4 -UHGTXR_E2E_PATCH_PAR -DHGTXR_E2E_PATCH_PAR=32 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -UHGTXR_E2E_MLP_TOKEN_PAR -DHGTXR_E2E_MLP_TOKEN_PAR=2 -UHGTXR_E2E_DENSE_TOKEN_PAR -DHGTXR_E2E_DENSE_TOKEN_PAR=4 -DHGTXR_E2E_MLP_FUSED_W2=1 -DHGTXR_E2E_MLP_FUSED_W2_BANKED_ACC=1 -DHGTXR_E2E_MLP_FUSED_W2_HP_PAR=2 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -UHGTXR_E2E_TEST_RUNTIME_MODES -DHGTXR_E2E_TEST_RUNTIME_MODES=0 -DHGTXR_E2E_FORCE_RUNTIME_MODE=0 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_mtok2_dtok4_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_track_only)
    PROJECT="hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_dtok4_300_mem16_track_only_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_URAM_ATTN -DHGTXR_E2E_URAM_ATTN=0 -UHGTXR_E2E_LUTRAM_ATTN -DHGTXR_E2E_LUTRAM_ATTN=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=4 -UHGTXR_E2E_PATCH_PAR -DHGTXR_E2E_PATCH_PAR=32 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -UHGTXR_E2E_MLP_TOKEN_PAR -DHGTXR_E2E_MLP_TOKEN_PAR=2 -UHGTXR_E2E_DENSE_TOKEN_PAR -DHGTXR_E2E_DENSE_TOKEN_PAR=4 -DHGTXR_E2E_MLP_FUSED_W2=1 -DHGTXR_E2E_MLP_FUSED_W2_BANKED_ACC=1 -DHGTXR_E2E_MLP_FUSED_W2_HP_PAR=2 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -UHGTXR_E2E_TEST_RUNTIME_MODES -DHGTXR_E2E_TEST_RUNTIME_MODES=0 -DHGTXR_E2E_FORCE_RUNTIME_MODE=1 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_mtok4_dtok4_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16)
    PROJECT="hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_300_mem16_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_URAM_ATTN -DHGTXR_E2E_URAM_ATTN=0 -UHGTXR_E2E_LUTRAM_ATTN -DHGTXR_E2E_LUTRAM_ATTN=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=4 -UHGTXR_E2E_PATCH_PAR -DHGTXR_E2E_PATCH_PAR=32 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -UHGTXR_E2E_MLP_TOKEN_PAR -DHGTXR_E2E_MLP_TOKEN_PAR=4 -UHGTXR_E2E_DENSE_TOKEN_PAR -DHGTXR_E2E_DENSE_TOKEN_PAR=4 -DHGTXR_E2E_MLP_FUSED_W2=1 -DHGTXR_E2E_MLP_FUSED_W2_BANKED_ACC=1 -DHGTXR_E2E_MLP_FUSED_W2_HP_PAR=2 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_mtok4_dtok4_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only)
    PROJECT="hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_300_mem16_search_only_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_URAM_ATTN -DHGTXR_E2E_URAM_ATTN=0 -UHGTXR_E2E_LUTRAM_ATTN -DHGTXR_E2E_LUTRAM_ATTN=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=4 -UHGTXR_E2E_PATCH_PAR -DHGTXR_E2E_PATCH_PAR=32 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -UHGTXR_E2E_MLP_TOKEN_PAR -DHGTXR_E2E_MLP_TOKEN_PAR=4 -UHGTXR_E2E_DENSE_TOKEN_PAR -DHGTXR_E2E_DENSE_TOKEN_PAR=4 -DHGTXR_E2E_MLP_FUSED_W2=1 -DHGTXR_E2E_MLP_FUSED_W2_BANKED_ACC=1 -DHGTXR_E2E_MLP_FUSED_W2_HP_PAR=2 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -UHGTXR_E2E_TEST_RUNTIME_MODES -DHGTXR_E2E_TEST_RUNTIME_MODES=0 -DHGTXR_E2E_FORCE_RUNTIME_MODE=0 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_mtok4_dtok4_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_track_only)
    PROJECT="hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_300_mem16_track_only_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_URAM_ATTN -DHGTXR_E2E_URAM_ATTN=0 -UHGTXR_E2E_LUTRAM_ATTN -DHGTXR_E2E_LUTRAM_ATTN=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=4 -UHGTXR_E2E_PATCH_PAR -DHGTXR_E2E_PATCH_PAR=32 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -UHGTXR_E2E_MLP_TOKEN_PAR -DHGTXR_E2E_MLP_TOKEN_PAR=4 -UHGTXR_E2E_DENSE_TOKEN_PAR -DHGTXR_E2E_DENSE_TOKEN_PAR=4 -DHGTXR_E2E_MLP_FUSED_W2=1 -DHGTXR_E2E_MLP_FUSED_W2_BANKED_ACC=1 -DHGTXR_E2E_MLP_FUSED_W2_HP_PAR=2 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -UHGTXR_E2E_TEST_RUNTIME_MODES -DHGTXR_E2E_TEST_RUNTIME_MODES=0 -DHGTXR_E2E_FORCE_RUNTIME_MODE=1 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_mtok4_dtok4_aq2_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16)
    PROJECT="hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq2_300_mem16_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_URAM_ATTN -DHGTXR_E2E_URAM_ATTN=0 -UHGTXR_E2E_LUTRAM_ATTN -DHGTXR_E2E_LUTRAM_ATTN=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=4 -UHGTXR_E2E_PATCH_PAR -DHGTXR_E2E_PATCH_PAR=32 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -UHGTXR_E2E_MLP_TOKEN_PAR -DHGTXR_E2E_MLP_TOKEN_PAR=4 -UHGTXR_E2E_DENSE_TOKEN_PAR -DHGTXR_E2E_DENSE_TOKEN_PAR=4 -UHGTXR_E2E_ATTN_QUERY_PAR -DHGTXR_E2E_ATTN_QUERY_PAR=2 -DHGTXR_E2E_MLP_FUSED_W2=1 -DHGTXR_E2E_MLP_FUSED_W2_BANKED_ACC=1 -DHGTXR_E2E_MLP_FUSED_W2_HP_PAR=2 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_mtok4_dtok4_aq2_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only)
    PROJECT="hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq2_300_mem16_search_only_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_URAM_ATTN -DHGTXR_E2E_URAM_ATTN=0 -UHGTXR_E2E_LUTRAM_ATTN -DHGTXR_E2E_LUTRAM_ATTN=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=4 -UHGTXR_E2E_PATCH_PAR -DHGTXR_E2E_PATCH_PAR=32 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -UHGTXR_E2E_MLP_TOKEN_PAR -DHGTXR_E2E_MLP_TOKEN_PAR=4 -UHGTXR_E2E_DENSE_TOKEN_PAR -DHGTXR_E2E_DENSE_TOKEN_PAR=4 -UHGTXR_E2E_ATTN_QUERY_PAR -DHGTXR_E2E_ATTN_QUERY_PAR=2 -DHGTXR_E2E_MLP_FUSED_W2=1 -DHGTXR_E2E_MLP_FUSED_W2_BANKED_ACC=1 -DHGTXR_E2E_MLP_FUSED_W2_HP_PAR=2 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -UHGTXR_E2E_TEST_RUNTIME_MODES -DHGTXR_E2E_TEST_RUNTIME_MODES=0 -DHGTXR_E2E_FORCE_RUNTIME_MODE=0 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_mtok4_dtok4_aq2_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_track_only)
    PROJECT="hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq2_300_mem16_track_only_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_URAM_ATTN -DHGTXR_E2E_URAM_ATTN=0 -UHGTXR_E2E_LUTRAM_ATTN -DHGTXR_E2E_LUTRAM_ATTN=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=4 -UHGTXR_E2E_PATCH_PAR -DHGTXR_E2E_PATCH_PAR=32 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -UHGTXR_E2E_MLP_TOKEN_PAR -DHGTXR_E2E_MLP_TOKEN_PAR=4 -UHGTXR_E2E_DENSE_TOKEN_PAR -DHGTXR_E2E_DENSE_TOKEN_PAR=4 -UHGTXR_E2E_ATTN_QUERY_PAR -DHGTXR_E2E_ATTN_QUERY_PAR=2 -DHGTXR_E2E_MLP_FUSED_W2=1 -DHGTXR_E2E_MLP_FUSED_W2_BANKED_ACC=1 -DHGTXR_E2E_MLP_FUSED_W2_HP_PAR=2 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -UHGTXR_E2E_TEST_RUNTIME_MODES -DHGTXR_E2E_TEST_RUNTIME_MODES=0 -DHGTXR_E2E_FORCE_RUNTIME_MODE=1 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_mtok4_dtok4_aq2_skiph_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16)
    PROJECT="hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq2_skiph_300_mem16_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_URAM_ATTN -DHGTXR_E2E_URAM_ATTN=0 -UHGTXR_E2E_LUTRAM_ATTN -DHGTXR_E2E_LUTRAM_ATTN=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=4 -UHGTXR_E2E_PATCH_PAR -DHGTXR_E2E_PATCH_PAR=32 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -UHGTXR_E2E_MLP_TOKEN_PAR -DHGTXR_E2E_MLP_TOKEN_PAR=4 -UHGTXR_E2E_DENSE_TOKEN_PAR -DHGTXR_E2E_DENSE_TOKEN_PAR=4 -UHGTXR_E2E_ATTN_QUERY_PAR -DHGTXR_E2E_ATTN_QUERY_PAR=2 -DHGTXR_E2E_MLP_FUSED_W2=1 -DHGTXR_E2E_MLP_FUSED_W2_BANKED_ACC=1 -DHGTXR_E2E_MLP_FUSED_W2_SKIP_HIDDEN_STORE=1 -DHGTXR_E2E_MLP_FUSED_W2_HP_PAR=2 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_mtok4_dtok4_aq2_skiph_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only)
    PROJECT="hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq2_skiph_300_mem16_search_only_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_URAM_ATTN -DHGTXR_E2E_URAM_ATTN=0 -UHGTXR_E2E_LUTRAM_ATTN -DHGTXR_E2E_LUTRAM_ATTN=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=4 -UHGTXR_E2E_PATCH_PAR -DHGTXR_E2E_PATCH_PAR=32 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -UHGTXR_E2E_MLP_TOKEN_PAR -DHGTXR_E2E_MLP_TOKEN_PAR=4 -UHGTXR_E2E_DENSE_TOKEN_PAR -DHGTXR_E2E_DENSE_TOKEN_PAR=4 -UHGTXR_E2E_ATTN_QUERY_PAR -DHGTXR_E2E_ATTN_QUERY_PAR=2 -DHGTXR_E2E_MLP_FUSED_W2=1 -DHGTXR_E2E_MLP_FUSED_W2_BANKED_ACC=1 -DHGTXR_E2E_MLP_FUSED_W2_SKIP_HIDDEN_STORE=1 -DHGTXR_E2E_MLP_FUSED_W2_HP_PAR=2 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -UHGTXR_E2E_TEST_RUNTIME_MODES -DHGTXR_E2E_TEST_RUNTIME_MODES=0 -DHGTXR_E2E_FORCE_RUNTIME_MODE=0 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_mtok4_dtok4_aq2_skiph_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_track_only)
    PROJECT="hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq2_skiph_300_mem16_track_only_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_URAM_ATTN -DHGTXR_E2E_URAM_ATTN=0 -UHGTXR_E2E_LUTRAM_ATTN -DHGTXR_E2E_LUTRAM_ATTN=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=4 -UHGTXR_E2E_PATCH_PAR -DHGTXR_E2E_PATCH_PAR=32 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -UHGTXR_E2E_MLP_TOKEN_PAR -DHGTXR_E2E_MLP_TOKEN_PAR=4 -UHGTXR_E2E_DENSE_TOKEN_PAR -DHGTXR_E2E_DENSE_TOKEN_PAR=4 -UHGTXR_E2E_ATTN_QUERY_PAR -DHGTXR_E2E_ATTN_QUERY_PAR=2 -DHGTXR_E2E_MLP_FUSED_W2=1 -DHGTXR_E2E_MLP_FUSED_W2_BANKED_ACC=1 -DHGTXR_E2E_MLP_FUSED_W2_SKIP_HIDDEN_STORE=1 -DHGTXR_E2E_MLP_FUSED_W2_HP_PAR=2 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -UHGTXR_E2E_TEST_RUNTIME_MODES -DHGTXR_E2E_TEST_RUNTIME_MODES=0 -DHGTXR_E2E_FORCE_RUNTIME_MODE=1 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar4_mtok4_dtok4_aq2_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only)
    PROJECT="hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq2_hp4_300_mem16_search_only_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_URAM_ATTN -DHGTXR_E2E_URAM_ATTN=0 -UHGTXR_E2E_LUTRAM_ATTN -DHGTXR_E2E_LUTRAM_ATTN=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=4 -UHGTXR_E2E_PATCH_PAR -DHGTXR_E2E_PATCH_PAR=32 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -UHGTXR_E2E_MLP_TOKEN_PAR -DHGTXR_E2E_MLP_TOKEN_PAR=4 -UHGTXR_E2E_DENSE_TOKEN_PAR -DHGTXR_E2E_DENSE_TOKEN_PAR=4 -UHGTXR_E2E_ATTN_QUERY_PAR -DHGTXR_E2E_ATTN_QUERY_PAR=2 -DHGTXR_E2E_MLP_FUSED_W2=1 -DHGTXR_E2E_MLP_FUSED_W2_BANKED_ACC=1 -DHGTXR_E2E_MLP_FUSED_W2_HP_PAR=4 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -UHGTXR_E2E_TEST_RUNTIME_MODES -DHGTXR_E2E_TEST_RUNTIME_MODES=0 -DHGTXR_E2E_FORCE_RUNTIME_MODE=0 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar4_w2hbank_mtok4_dtok4_aq2_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only)
    PROJECT="hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq2_hp4_w2hbank_300_mem16_search_only_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_URAM_ATTN -DHGTXR_E2E_URAM_ATTN=0 -UHGTXR_E2E_LUTRAM_ATTN -DHGTXR_E2E_LUTRAM_ATTN=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=4 -UHGTXR_E2E_PATCH_PAR -DHGTXR_E2E_PATCH_PAR=32 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -UHGTXR_E2E_MLP_TOKEN_PAR -DHGTXR_E2E_MLP_TOKEN_PAR=4 -UHGTXR_E2E_DENSE_TOKEN_PAR -DHGTXR_E2E_DENSE_TOKEN_PAR=4 -UHGTXR_E2E_ATTN_QUERY_PAR -DHGTXR_E2E_ATTN_QUERY_PAR=2 -DHGTXR_E2E_MLP_FUSED_W2=1 -DHGTXR_E2E_MLP_FUSED_W2_BANKED_ACC=1 -DHGTXR_E2E_MLP_FUSED_W2_HP_PAR=4 -DHGTXR_E2E_MLP_W2_HIDDEN_BANK_CACHE=1 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -UHGTXR_E2E_TEST_RUNTIME_MODES -DHGTXR_E2E_TEST_RUNTIME_MODES=0 -DHGTXR_E2E_FORCE_RUNTIME_MODE=0 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar4_w2hgroup_mtok4_dtok4_aq2_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only)
    PROJECT="hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq2_hp4_w2hgroup_300_mem16_search_only_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_URAM_ATTN -DHGTXR_E2E_URAM_ATTN=0 -UHGTXR_E2E_LUTRAM_ATTN -DHGTXR_E2E_LUTRAM_ATTN=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=4 -UHGTXR_E2E_PATCH_PAR -DHGTXR_E2E_PATCH_PAR=32 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -UHGTXR_E2E_MLP_TOKEN_PAR -DHGTXR_E2E_MLP_TOKEN_PAR=4 -UHGTXR_E2E_DENSE_TOKEN_PAR -DHGTXR_E2E_DENSE_TOKEN_PAR=4 -UHGTXR_E2E_ATTN_QUERY_PAR -DHGTXR_E2E_ATTN_QUERY_PAR=2 -DHGTXR_E2E_MLP_FUSED_W2=1 -DHGTXR_E2E_MLP_FUSED_W2_BANKED_ACC=1 -DHGTXR_E2E_MLP_FUSED_W2_HP_PAR=4 -DHGTXR_E2E_MLP_W2_HGROUP_BANK_CACHE=1 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -UHGTXR_E2E_TEST_RUNTIME_MODES -DHGTXR_E2E_TEST_RUNTIME_MODES=0 -DHGTXR_E2E_FORCE_RUNTIME_MODE=0 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar4_w2hgroup_mtok4_dtok4_aq4_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only)
    PROJECT="hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq4_hp4_w2hgroup_300_mem16_search_only_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_URAM_ATTN -DHGTXR_E2E_URAM_ATTN=0 -UHGTXR_E2E_LUTRAM_ATTN -DHGTXR_E2E_LUTRAM_ATTN=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=4 -UHGTXR_E2E_PATCH_PAR -DHGTXR_E2E_PATCH_PAR=32 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -UHGTXR_E2E_MLP_TOKEN_PAR -DHGTXR_E2E_MLP_TOKEN_PAR=4 -UHGTXR_E2E_DENSE_TOKEN_PAR -DHGTXR_E2E_DENSE_TOKEN_PAR=4 -UHGTXR_E2E_ATTN_QUERY_PAR -DHGTXR_E2E_ATTN_QUERY_PAR=4 -DHGTXR_E2E_MLP_FUSED_W2=1 -DHGTXR_E2E_MLP_FUSED_W2_BANKED_ACC=1 -DHGTXR_E2E_MLP_FUSED_W2_HP_PAR=4 -DHGTXR_E2E_MLP_W2_HGROUP_BANK_CACHE=1 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -UHGTXR_E2E_TEST_RUNTIME_MODES -DHGTXR_E2E_TEST_RUNTIME_MODES=0 -DHGTXR_E2E_FORCE_RUNTIME_MODE=0 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar4_w2hgroup_mtok4_dtok4_aq4_qkvbram_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only)
    PROJECT="hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq4_hp4_w2hgroup_qkvbram_300_mem16_search_only_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_URAM_Q -DHGTXR_E2E_URAM_Q=0 -UHGTXR_E2E_LUTRAM_Q -DHGTXR_E2E_LUTRAM_Q=0 -UHGTXR_E2E_URAM_K -DHGTXR_E2E_URAM_K=0 -UHGTXR_E2E_LUTRAM_K -DHGTXR_E2E_LUTRAM_K=0 -UHGTXR_E2E_URAM_V -DHGTXR_E2E_URAM_V=0 -UHGTXR_E2E_LUTRAM_V -DHGTXR_E2E_LUTRAM_V=0 -UHGTXR_E2E_URAM_ATTN -DHGTXR_E2E_URAM_ATTN=0 -UHGTXR_E2E_LUTRAM_ATTN -DHGTXR_E2E_LUTRAM_ATTN=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=4 -UHGTXR_E2E_PATCH_PAR -DHGTXR_E2E_PATCH_PAR=32 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -UHGTXR_E2E_MLP_TOKEN_PAR -DHGTXR_E2E_MLP_TOKEN_PAR=4 -UHGTXR_E2E_DENSE_TOKEN_PAR -DHGTXR_E2E_DENSE_TOKEN_PAR=4 -UHGTXR_E2E_ATTN_QUERY_PAR -DHGTXR_E2E_ATTN_QUERY_PAR=4 -DHGTXR_E2E_MLP_FUSED_W2=1 -DHGTXR_E2E_MLP_FUSED_W2_BANKED_ACC=1 -DHGTXR_E2E_MLP_FUSED_W2_HP_PAR=4 -DHGTXR_E2E_MLP_W2_HGROUP_BANK_CACHE=1 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -UHGTXR_E2E_TEST_RUNTIME_MODES -DHGTXR_E2E_TEST_RUNTIME_MODES=0 -DHGTXR_E2E_FORCE_RUNTIME_MODE=0 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar4_w2hgroup_mtok4_dtok4_aq4_qkvbram_tail2_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only)
    PROJECT="hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq4_hp4_w2hgroup_qkvbram_tail2_300_mem16_search_only_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_URAM_Q -DHGTXR_E2E_URAM_Q=0 -UHGTXR_E2E_LUTRAM_Q -DHGTXR_E2E_LUTRAM_Q=0 -UHGTXR_E2E_URAM_K -DHGTXR_E2E_URAM_K=0 -UHGTXR_E2E_LUTRAM_K -DHGTXR_E2E_LUTRAM_K=0 -UHGTXR_E2E_URAM_V -DHGTXR_E2E_URAM_V=0 -UHGTXR_E2E_LUTRAM_V -DHGTXR_E2E_LUTRAM_V=0 -UHGTXR_E2E_URAM_ATTN -DHGTXR_E2E_URAM_ATTN=0 -UHGTXR_E2E_LUTRAM_ATTN -DHGTXR_E2E_LUTRAM_ATTN=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_CORE_FABRIC_TAIL_LANES -DHGTXR_E2E_CORE_FABRIC_TAIL_LANES=2 -UHGTXR_E2E_CORE_LANE_CT_SWITCH -DHGTXR_E2E_CORE_LANE_CT_SWITCH=1 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=4 -UHGTXR_E2E_PATCH_PAR -DHGTXR_E2E_PATCH_PAR=32 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -UHGTXR_E2E_MLP_TOKEN_PAR -DHGTXR_E2E_MLP_TOKEN_PAR=4 -UHGTXR_E2E_DENSE_TOKEN_PAR -DHGTXR_E2E_DENSE_TOKEN_PAR=4 -UHGTXR_E2E_ATTN_QUERY_PAR -DHGTXR_E2E_ATTN_QUERY_PAR=4 -DHGTXR_E2E_MLP_FUSED_W2=1 -DHGTXR_E2E_MLP_FUSED_W2_BANKED_ACC=1 -DHGTXR_E2E_MLP_FUSED_W2_HP_PAR=4 -DHGTXR_E2E_MLP_W2_HGROUP_BANK_CACHE=1 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -UHGTXR_E2E_TEST_RUNTIME_MODES -DHGTXR_E2E_TEST_RUNTIME_MODES=0 -DHGTXR_E2E_FORCE_RUNTIME_MODE=0 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar4_w2hgroup_mtok4_dtok4_aq4_qkvbram_tail2_patch64_attntok2_attnbram_nocache_patchpar64_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only)
    PROJECT="hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch64_mtok4_dtok4_aq4_hp4_w2hgroup_qkvbram_tail2_300_mem16_search_only_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_URAM_Q -DHGTXR_E2E_URAM_Q=0 -UHGTXR_E2E_LUTRAM_Q -DHGTXR_E2E_LUTRAM_Q=0 -UHGTXR_E2E_URAM_K -DHGTXR_E2E_URAM_K=0 -UHGTXR_E2E_LUTRAM_K -DHGTXR_E2E_LUTRAM_K=0 -UHGTXR_E2E_URAM_V -DHGTXR_E2E_URAM_V=0 -UHGTXR_E2E_LUTRAM_V -DHGTXR_E2E_LUTRAM_V=0 -UHGTXR_E2E_URAM_ATTN -DHGTXR_E2E_URAM_ATTN=0 -UHGTXR_E2E_LUTRAM_ATTN -DHGTXR_E2E_LUTRAM_ATTN=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_CORE_FABRIC_TAIL_LANES -DHGTXR_E2E_CORE_FABRIC_TAIL_LANES=2 -UHGTXR_E2E_CORE_LANE_CT_SWITCH -DHGTXR_E2E_CORE_LANE_CT_SWITCH=1 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=4 -UHGTXR_E2E_PATCH_PAR -DHGTXR_E2E_PATCH_PAR=64 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -UHGTXR_E2E_MLP_TOKEN_PAR -DHGTXR_E2E_MLP_TOKEN_PAR=4 -UHGTXR_E2E_DENSE_TOKEN_PAR -DHGTXR_E2E_DENSE_TOKEN_PAR=4 -UHGTXR_E2E_ATTN_QUERY_PAR -DHGTXR_E2E_ATTN_QUERY_PAR=4 -DHGTXR_E2E_MLP_FUSED_W2=1 -DHGTXR_E2E_MLP_FUSED_W2_BANKED_ACC=1 -DHGTXR_E2E_MLP_FUSED_W2_HP_PAR=4 -DHGTXR_E2E_MLP_W2_HGROUP_BANK_CACHE=1 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -UHGTXR_E2E_TEST_RUNTIME_MODES -DHGTXR_E2E_TEST_RUNTIME_MODES=0 -DHGTXR_E2E_FORCE_RUNTIME_MODE=0 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar4_w2hgroup_mtok4_dtok4_aq4_qkvbram_tail2_exppart_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only)
    PROJECT="hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq4_hp4_w2hgroup_qkvbram_tail2_exppart_300_mem16_search_only_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_URAM_Q -DHGTXR_E2E_URAM_Q=0 -UHGTXR_E2E_LUTRAM_Q -DHGTXR_E2E_LUTRAM_Q=0 -UHGTXR_E2E_URAM_K -DHGTXR_E2E_URAM_K=0 -UHGTXR_E2E_LUTRAM_K -DHGTXR_E2E_LUTRAM_K=0 -UHGTXR_E2E_URAM_V -DHGTXR_E2E_URAM_V=0 -UHGTXR_E2E_LUTRAM_V -DHGTXR_E2E_LUTRAM_V=0 -UHGTXR_E2E_URAM_ATTN -DHGTXR_E2E_URAM_ATTN=0 -UHGTXR_E2E_LUTRAM_ATTN -DHGTXR_E2E_LUTRAM_ATTN=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_CORE_FABRIC_TAIL_LANES -DHGTXR_E2E_CORE_FABRIC_TAIL_LANES=2 -UHGTXR_E2E_CORE_LANE_CT_SWITCH -DHGTXR_E2E_CORE_LANE_CT_SWITCH=1 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=4 -UHGTXR_E2E_PATCH_PAR -DHGTXR_E2E_PATCH_PAR=32 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -UHGTXR_E2E_MLP_TOKEN_PAR -DHGTXR_E2E_MLP_TOKEN_PAR=4 -UHGTXR_E2E_DENSE_TOKEN_PAR -DHGTXR_E2E_DENSE_TOKEN_PAR=4 -UHGTXR_E2E_ATTN_QUERY_PAR -DHGTXR_E2E_ATTN_QUERY_PAR=4 -DHGTXR_E2E_MLP_FUSED_W2=1 -DHGTXR_E2E_MLP_FUSED_W2_BANKED_ACC=1 -DHGTXR_E2E_MLP_FUSED_W2_HP_PAR=4 -DHGTXR_E2E_MLP_W2_HGROUP_BANK_CACHE=1 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_SOFTMAX_EXP_ROM_PARTITION=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -UHGTXR_E2E_TEST_RUNTIME_MODES -DHGTXR_E2E_TEST_RUNTIME_MODES=0 -DHGTXR_E2E_FORCE_RUNTIME_MODE=0 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar4_w2hgroup_mtok4_dtok4_aq4_qkvbram_tail2_exppart_skiph_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only)
    PROJECT="hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq4_hp4_w2hgroup_qkvbram_tail2_exppart_skiph_300_mem16_search_only_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_URAM_Q -DHGTXR_E2E_URAM_Q=0 -UHGTXR_E2E_LUTRAM_Q -DHGTXR_E2E_LUTRAM_Q=0 -UHGTXR_E2E_URAM_K -DHGTXR_E2E_URAM_K=0 -UHGTXR_E2E_LUTRAM_K -DHGTXR_E2E_LUTRAM_K=0 -UHGTXR_E2E_URAM_V -DHGTXR_E2E_URAM_V=0 -UHGTXR_E2E_LUTRAM_V -DHGTXR_E2E_LUTRAM_V=0 -UHGTXR_E2E_URAM_ATTN -DHGTXR_E2E_URAM_ATTN=0 -UHGTXR_E2E_LUTRAM_ATTN -DHGTXR_E2E_LUTRAM_ATTN=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_CORE_FABRIC_TAIL_LANES -DHGTXR_E2E_CORE_FABRIC_TAIL_LANES=2 -UHGTXR_E2E_CORE_LANE_CT_SWITCH -DHGTXR_E2E_CORE_LANE_CT_SWITCH=1 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=4 -UHGTXR_E2E_PATCH_PAR -DHGTXR_E2E_PATCH_PAR=32 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -UHGTXR_E2E_MLP_TOKEN_PAR -DHGTXR_E2E_MLP_TOKEN_PAR=4 -UHGTXR_E2E_DENSE_TOKEN_PAR -DHGTXR_E2E_DENSE_TOKEN_PAR=4 -UHGTXR_E2E_ATTN_QUERY_PAR -DHGTXR_E2E_ATTN_QUERY_PAR=4 -DHGTXR_E2E_MLP_FUSED_W2=1 -DHGTXR_E2E_MLP_FUSED_W2_BANKED_ACC=1 -DHGTXR_E2E_MLP_FUSED_W2_HP_PAR=4 -DHGTXR_E2E_MLP_FUSED_W2_SKIP_HIDDEN_STORE=1 -DHGTXR_E2E_MLP_W2_HGROUP_BANK_CACHE=1 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_SOFTMAX_EXP_ROM_PARTITION=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -UHGTXR_E2E_TEST_RUNTIME_MODES -DHGTXR_E2E_TEST_RUNTIME_MODES=0 -DHGTXR_E2E_FORCE_RUNTIME_MODE=0 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar4_w2hgroup_mtok4_dtok4_aq4_qkvbram_tail2_exppart_w1c2_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only)
    PROJECT="hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq4_hp4_w2hgroup_qkvbram_tail2_exppart_w1c2_300_mem16_search_only_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_URAM_Q -DHGTXR_E2E_URAM_Q=0 -UHGTXR_E2E_LUTRAM_Q -DHGTXR_E2E_LUTRAM_Q=0 -UHGTXR_E2E_URAM_K -DHGTXR_E2E_URAM_K=0 -UHGTXR_E2E_LUTRAM_K -DHGTXR_E2E_LUTRAM_K=0 -UHGTXR_E2E_URAM_V -DHGTXR_E2E_URAM_V=0 -UHGTXR_E2E_LUTRAM_V -DHGTXR_E2E_LUTRAM_V=0 -UHGTXR_E2E_URAM_ATTN -DHGTXR_E2E_URAM_ATTN=0 -UHGTXR_E2E_LUTRAM_ATTN -DHGTXR_E2E_LUTRAM_ATTN=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_CORE_FABRIC_TAIL_LANES -DHGTXR_E2E_CORE_FABRIC_TAIL_LANES=2 -UHGTXR_E2E_CORE_LANE_CT_SWITCH -DHGTXR_E2E_CORE_LANE_CT_SWITCH=1 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=4 -UHGTXR_E2E_PATCH_PAR -DHGTXR_E2E_PATCH_PAR=32 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -UHGTXR_E2E_MLP_TOKEN_PAR -DHGTXR_E2E_MLP_TOKEN_PAR=4 -UHGTXR_E2E_DENSE_TOKEN_PAR -DHGTXR_E2E_DENSE_TOKEN_PAR=4 -UHGTXR_E2E_ATTN_QUERY_PAR -DHGTXR_E2E_ATTN_QUERY_PAR=4 -DHGTXR_E2E_MLP_FUSED_W2=1 -DHGTXR_E2E_MLP_FUSED_W2_BANKED_ACC=1 -DHGTXR_E2E_MLP_FUSED_W2_HP_PAR=4 -DHGTXR_E2E_MLP_W1_C_PAR=2 -DHGTXR_E2E_MLP_W2_HGROUP_BANK_CACHE=1 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_SOFTMAX_EXP_ROM_PARTITION=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -UHGTXR_E2E_TEST_RUNTIME_MODES -DHGTXR_E2E_TEST_RUNTIME_MODES=0 -DHGTXR_E2E_FORCE_RUNTIME_MODE=0 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar4_w2hgroup_mtok4_dtok4_aq4_qkvbram_tail2_exppart_w1c3_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only)
    PROJECT="hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq4_hp4_w2hgroup_qkvbram_tail2_exppart_w1c3_300_mem16_search_only_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_URAM_Q -DHGTXR_E2E_URAM_Q=0 -UHGTXR_E2E_LUTRAM_Q -DHGTXR_E2E_LUTRAM_Q=0 -UHGTXR_E2E_URAM_K -DHGTXR_E2E_URAM_K=0 -UHGTXR_E2E_LUTRAM_K -DHGTXR_E2E_LUTRAM_K=0 -UHGTXR_E2E_URAM_V -DHGTXR_E2E_URAM_V=0 -UHGTXR_E2E_LUTRAM_V -DHGTXR_E2E_LUTRAM_V=0 -UHGTXR_E2E_URAM_ATTN -DHGTXR_E2E_URAM_ATTN=0 -UHGTXR_E2E_LUTRAM_ATTN -DHGTXR_E2E_LUTRAM_ATTN=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_CORE_FABRIC_TAIL_LANES -DHGTXR_E2E_CORE_FABRIC_TAIL_LANES=2 -UHGTXR_E2E_CORE_LANE_CT_SWITCH -DHGTXR_E2E_CORE_LANE_CT_SWITCH=1 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=4 -UHGTXR_E2E_PATCH_PAR -DHGTXR_E2E_PATCH_PAR=32 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -UHGTXR_E2E_MLP_TOKEN_PAR -DHGTXR_E2E_MLP_TOKEN_PAR=4 -UHGTXR_E2E_DENSE_TOKEN_PAR -DHGTXR_E2E_DENSE_TOKEN_PAR=4 -UHGTXR_E2E_ATTN_QUERY_PAR -DHGTXR_E2E_ATTN_QUERY_PAR=4 -DHGTXR_E2E_MLP_FUSED_W2=1 -DHGTXR_E2E_MLP_FUSED_W2_BANKED_ACC=1 -DHGTXR_E2E_MLP_FUSED_W2_HP_PAR=4 -DHGTXR_E2E_MLP_W1_C_PAR=3 -DHGTXR_E2E_MLP_W2_HGROUP_BANK_CACHE=1 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_SOFTMAX_EXP_ROM_PARTITION=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -UHGTXR_E2E_TEST_RUNTIME_MODES -DHGTXR_E2E_TEST_RUNTIME_MODES=0 -DHGTXR_E2E_FORCE_RUNTIME_MODE=0 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar4_w2hgroup_mtok4_dtok4_aq8_qkvbram_tail2_exppart_w1c2_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only)
    PROJECT="hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq8_hp4_w2hgroup_qkvbram_tail2_exppart_w1c2_300_mem16_search_only_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_URAM_Q -DHGTXR_E2E_URAM_Q=0 -UHGTXR_E2E_LUTRAM_Q -DHGTXR_E2E_LUTRAM_Q=0 -UHGTXR_E2E_URAM_K -DHGTXR_E2E_URAM_K=0 -UHGTXR_E2E_LUTRAM_K -DHGTXR_E2E_LUTRAM_K=0 -UHGTXR_E2E_URAM_V -DHGTXR_E2E_URAM_V=0 -UHGTXR_E2E_LUTRAM_V -DHGTXR_E2E_LUTRAM_V=0 -UHGTXR_E2E_URAM_ATTN -DHGTXR_E2E_URAM_ATTN=0 -UHGTXR_E2E_LUTRAM_ATTN -DHGTXR_E2E_LUTRAM_ATTN=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_CORE_FABRIC_TAIL_LANES -DHGTXR_E2E_CORE_FABRIC_TAIL_LANES=2 -UHGTXR_E2E_CORE_LANE_CT_SWITCH -DHGTXR_E2E_CORE_LANE_CT_SWITCH=1 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=4 -UHGTXR_E2E_PATCH_PAR -DHGTXR_E2E_PATCH_PAR=32 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -UHGTXR_E2E_MLP_TOKEN_PAR -DHGTXR_E2E_MLP_TOKEN_PAR=4 -UHGTXR_E2E_DENSE_TOKEN_PAR -DHGTXR_E2E_DENSE_TOKEN_PAR=4 -UHGTXR_E2E_ATTN_QUERY_PAR -DHGTXR_E2E_ATTN_QUERY_PAR=8 -DHGTXR_E2E_MLP_FUSED_W2=1 -DHGTXR_E2E_MLP_FUSED_W2_BANKED_ACC=1 -DHGTXR_E2E_MLP_FUSED_W2_HP_PAR=4 -DHGTXR_E2E_MLP_W1_C_PAR=2 -DHGTXR_E2E_MLP_W2_HGROUP_BANK_CACHE=1 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_SOFTMAX_EXP_ROM_PARTITION=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -UHGTXR_E2E_TEST_RUNTIME_MODES -DHGTXR_E2E_TEST_RUNTIME_MODES=0 -DHGTXR_E2E_FORCE_RUNTIME_MODE=0 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar8_mlpfusebank_hpar8_w2hgroup_mtok4_dtok4_aq8_qkvbram_tail2_exppart_w1c2_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only)
    PROJECT="hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq8_hp8_w2hgroup_qkvbram_tail2_exppart_w1c2_300_mem16_search_only_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_URAM_Q -DHGTXR_E2E_URAM_Q=0 -UHGTXR_E2E_LUTRAM_Q -DHGTXR_E2E_LUTRAM_Q=0 -UHGTXR_E2E_URAM_K -DHGTXR_E2E_URAM_K=0 -UHGTXR_E2E_LUTRAM_K -DHGTXR_E2E_LUTRAM_K=0 -UHGTXR_E2E_URAM_V -DHGTXR_E2E_URAM_V=0 -UHGTXR_E2E_LUTRAM_V -DHGTXR_E2E_LUTRAM_V=0 -UHGTXR_E2E_URAM_ATTN -DHGTXR_E2E_URAM_ATTN=0 -UHGTXR_E2E_LUTRAM_ATTN -DHGTXR_E2E_LUTRAM_ATTN=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_CORE_FABRIC_TAIL_LANES -DHGTXR_E2E_CORE_FABRIC_TAIL_LANES=2 -UHGTXR_E2E_CORE_LANE_CT_SWITCH -DHGTXR_E2E_CORE_LANE_CT_SWITCH=1 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=8 -UHGTXR_E2E_PATCH_PAR -DHGTXR_E2E_PATCH_PAR=32 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -UHGTXR_E2E_MLP_TOKEN_PAR -DHGTXR_E2E_MLP_TOKEN_PAR=4 -UHGTXR_E2E_DENSE_TOKEN_PAR -DHGTXR_E2E_DENSE_TOKEN_PAR=4 -UHGTXR_E2E_ATTN_QUERY_PAR -DHGTXR_E2E_ATTN_QUERY_PAR=8 -DHGTXR_E2E_MLP_FUSED_W2=1 -DHGTXR_E2E_MLP_FUSED_W2_BANKED_ACC=1 -DHGTXR_E2E_MLP_FUSED_W2_HP_PAR=8 -DHGTXR_E2E_MLP_W1_C_PAR=2 -DHGTXR_E2E_MLP_W2_HGROUP_BANK_CACHE=1 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_SOFTMAX_EXP_ROM_PARTITION=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -UHGTXR_E2E_TEST_RUNTIME_MODES -DHGTXR_E2E_TEST_RUNTIME_MODES=0 -DHGTXR_E2E_FORCE_RUNTIME_MODE=0 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar8_mlpfusebank_hpar8_acc24_w2hgroup_mtok4_dtok4_aq8_qkvbram_tail2_exppart_w1c2_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only)
    PROJECT="hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq8_hp8_acc24_w2hgroup_qkvbram_tail2_exppart_w1c2_300_mem16_search_only_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_ACC_W -DHGTXR_ACC_W=24 -UHGTXR_ACC_I -DHGTXR_ACC_I=10 -UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_URAM_Q -DHGTXR_E2E_URAM_Q=0 -UHGTXR_E2E_LUTRAM_Q -DHGTXR_E2E_LUTRAM_Q=0 -UHGTXR_E2E_URAM_K -DHGTXR_E2E_URAM_K=0 -UHGTXR_E2E_LUTRAM_K -DHGTXR_E2E_LUTRAM_K=0 -UHGTXR_E2E_URAM_V -DHGTXR_E2E_URAM_V=0 -UHGTXR_E2E_LUTRAM_V -DHGTXR_E2E_LUTRAM_V=0 -UHGTXR_E2E_URAM_ATTN -DHGTXR_E2E_URAM_ATTN=0 -UHGTXR_E2E_LUTRAM_ATTN -DHGTXR_E2E_LUTRAM_ATTN=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_CORE_FABRIC_TAIL_LANES -DHGTXR_E2E_CORE_FABRIC_TAIL_LANES=2 -UHGTXR_E2E_CORE_LANE_CT_SWITCH -DHGTXR_E2E_CORE_LANE_CT_SWITCH=1 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=8 -UHGTXR_E2E_PATCH_PAR -DHGTXR_E2E_PATCH_PAR=32 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -UHGTXR_E2E_MLP_TOKEN_PAR -DHGTXR_E2E_MLP_TOKEN_PAR=4 -UHGTXR_E2E_DENSE_TOKEN_PAR -DHGTXR_E2E_DENSE_TOKEN_PAR=4 -UHGTXR_E2E_ATTN_QUERY_PAR -DHGTXR_E2E_ATTN_QUERY_PAR=8 -DHGTXR_E2E_MLP_FUSED_W2=1 -DHGTXR_E2E_MLP_FUSED_W2_BANKED_ACC=1 -DHGTXR_E2E_MLP_FUSED_W2_HP_PAR=8 -DHGTXR_E2E_MLP_W1_C_PAR=2 -DHGTXR_E2E_MLP_W2_HGROUP_BANK_CACHE=1 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_SOFTMAX_EXP_ROM_PARTITION=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -UHGTXR_E2E_TEST_RUNTIME_MODES -DHGTXR_E2E_TEST_RUNTIME_MODES=0 -DHGTXR_E2E_FORCE_RUNTIME_MODE=0 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar8_mlpfusebank_hpar16_acc24_w2hgroup_mtok4_dtok4_aq8_qkvbram_tail2_exppart_w1c2_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only)
    PROJECT="hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq8_hp16_acc24_w2hgroup_qkvbram_tail2_exppart_w1c2_300_mem16_search_only_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_ACC_W -DHGTXR_ACC_W=24 -UHGTXR_ACC_I -DHGTXR_ACC_I=10 -UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_URAM_Q -DHGTXR_E2E_URAM_Q=0 -UHGTXR_E2E_LUTRAM_Q -DHGTXR_E2E_LUTRAM_Q=0 -UHGTXR_E2E_URAM_K -DHGTXR_E2E_URAM_K=0 -UHGTXR_E2E_LUTRAM_K -DHGTXR_E2E_LUTRAM_K=0 -UHGTXR_E2E_URAM_V -DHGTXR_E2E_URAM_V=0 -UHGTXR_E2E_LUTRAM_V -DHGTXR_E2E_LUTRAM_V=0 -UHGTXR_E2E_URAM_ATTN -DHGTXR_E2E_URAM_ATTN=0 -UHGTXR_E2E_LUTRAM_ATTN -DHGTXR_E2E_LUTRAM_ATTN=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_CORE_FABRIC_TAIL_LANES -DHGTXR_E2E_CORE_FABRIC_TAIL_LANES=2 -UHGTXR_E2E_CORE_LANE_CT_SWITCH -DHGTXR_E2E_CORE_LANE_CT_SWITCH=1 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=8 -UHGTXR_E2E_PATCH_PAR -DHGTXR_E2E_PATCH_PAR=32 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -UHGTXR_E2E_MLP_TOKEN_PAR -DHGTXR_E2E_MLP_TOKEN_PAR=4 -UHGTXR_E2E_DENSE_TOKEN_PAR -DHGTXR_E2E_DENSE_TOKEN_PAR=4 -UHGTXR_E2E_ATTN_QUERY_PAR -DHGTXR_E2E_ATTN_QUERY_PAR=8 -DHGTXR_E2E_MLP_FUSED_W2=1 -DHGTXR_E2E_MLP_FUSED_W2_BANKED_ACC=1 -DHGTXR_E2E_MLP_FUSED_W2_HP_PAR=16 -DHGTXR_E2E_MLP_W1_C_PAR=2 -DHGTXR_E2E_MLP_W2_HGROUP_BANK_CACHE=1 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_SOFTMAX_EXP_ROM_PARTITION=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -UHGTXR_E2E_TEST_RUNTIME_MODES -DHGTXR_E2E_TEST_RUNTIME_MODES=0 -DHGTXR_E2E_FORCE_RUNTIME_MODE=0 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  goal_full_rom_scheduler_hpar16_gelupart_300_mem16)
    PROJECT="hgtxr_e2e_axis_goal_full_rom_scheduler_hpar16_gelupart_300_mem16_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_ACC_W -DHGTXR_ACC_W=24 -UHGTXR_ACC_I -DHGTXR_ACC_I=10 -UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_URAM_Q -DHGTXR_E2E_URAM_Q=0 -UHGTXR_E2E_LUTRAM_Q -DHGTXR_E2E_LUTRAM_Q=0 -UHGTXR_E2E_URAM_K -DHGTXR_E2E_URAM_K=0 -UHGTXR_E2E_LUTRAM_K -DHGTXR_E2E_LUTRAM_K=0 -UHGTXR_E2E_URAM_V -DHGTXR_E2E_URAM_V=0 -UHGTXR_E2E_LUTRAM_V -DHGTXR_E2E_LUTRAM_V=0 -UHGTXR_E2E_URAM_ATTN -DHGTXR_E2E_URAM_ATTN=0 -UHGTXR_E2E_LUTRAM_ATTN -DHGTXR_E2E_LUTRAM_ATTN=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_CORE_FABRIC_TAIL_LANES -DHGTXR_E2E_CORE_FABRIC_TAIL_LANES=2 -UHGTXR_E2E_CORE_LANE_CT_SWITCH -DHGTXR_E2E_CORE_LANE_CT_SWITCH=1 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=8 -UHGTXR_E2E_PATCH_PAR -DHGTXR_E2E_PATCH_PAR=32 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -UHGTXR_E2E_MLP_TOKEN_PAR -DHGTXR_E2E_MLP_TOKEN_PAR=4 -UHGTXR_E2E_DENSE_TOKEN_PAR -DHGTXR_E2E_DENSE_TOKEN_PAR=4 -UHGTXR_E2E_ATTN_QUERY_PAR -DHGTXR_E2E_ATTN_QUERY_PAR=8 -DHGTXR_E2E_MLP_FUSED_W2=1 -DHGTXR_E2E_MLP_FUSED_W2_BANKED_ACC=1 -DHGTXR_E2E_MLP_FUSED_W2_HP_PAR=16 -DHGTXR_E2E_MLP_W1_C_PAR=2 -DHGTXR_E2E_MLP_W2_HGROUP_BANK_CACHE=1 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_GELU_ROM_PARTITION=1 -DHGTXR_E2E_SOFTMAX_EXP_ROM_PARTITION=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  goal_full_rom_scheduler_hpar16_gelupart_300_mem16_search_only|par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar8_mlpfusebank_hpar16_acc24_gelupart_w2hgroup_mtok4_dtok4_aq8_qkvbram_tail2_exppart_w1c2_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only)
    PROJECT="hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq8_hp16_acc24_gelupart_w2hgroup_qkvbram_tail2_exppart_w1c2_300_mem16_search_only_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_ACC_W -DHGTXR_ACC_W=24 -UHGTXR_ACC_I -DHGTXR_ACC_I=10 -UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_URAM_Q -DHGTXR_E2E_URAM_Q=0 -UHGTXR_E2E_LUTRAM_Q -DHGTXR_E2E_LUTRAM_Q=0 -UHGTXR_E2E_URAM_K -DHGTXR_E2E_URAM_K=0 -UHGTXR_E2E_LUTRAM_K -DHGTXR_E2E_LUTRAM_K=0 -UHGTXR_E2E_URAM_V -DHGTXR_E2E_URAM_V=0 -UHGTXR_E2E_LUTRAM_V -DHGTXR_E2E_LUTRAM_V=0 -UHGTXR_E2E_URAM_ATTN -DHGTXR_E2E_URAM_ATTN=0 -UHGTXR_E2E_LUTRAM_ATTN -DHGTXR_E2E_LUTRAM_ATTN=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_CORE_FABRIC_TAIL_LANES -DHGTXR_E2E_CORE_FABRIC_TAIL_LANES=2 -UHGTXR_E2E_CORE_LANE_CT_SWITCH -DHGTXR_E2E_CORE_LANE_CT_SWITCH=1 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=8 -UHGTXR_E2E_PATCH_PAR -DHGTXR_E2E_PATCH_PAR=32 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -UHGTXR_E2E_MLP_TOKEN_PAR -DHGTXR_E2E_MLP_TOKEN_PAR=4 -UHGTXR_E2E_DENSE_TOKEN_PAR -DHGTXR_E2E_DENSE_TOKEN_PAR=4 -UHGTXR_E2E_ATTN_QUERY_PAR -DHGTXR_E2E_ATTN_QUERY_PAR=8 -DHGTXR_E2E_MLP_FUSED_W2=1 -DHGTXR_E2E_MLP_FUSED_W2_BANKED_ACC=1 -DHGTXR_E2E_MLP_FUSED_W2_HP_PAR=16 -DHGTXR_E2E_MLP_W1_C_PAR=2 -DHGTXR_E2E_MLP_W2_HGROUP_BANK_CACHE=1 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_GELU_ROM_PARTITION=1 -DHGTXR_E2E_SOFTMAX_EXP_ROM_PARTITION=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -UHGTXR_E2E_TEST_RUNTIME_MODES -DHGTXR_E2E_TEST_RUNTIME_MODES=0 -DHGTXR_E2E_FORCE_RUNTIME_MODE=0 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  goal_full_rom_scheduler_hpar16_gelupart_300_mem16_track_only)
    PROJECT="hgtxr_e2e_axis_goal_full_rom_scheduler_hpar16_gelupart_300_mem16_track_only_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_ACC_W -DHGTXR_ACC_W=24 -UHGTXR_ACC_I -DHGTXR_ACC_I=10 -UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_URAM_Q -DHGTXR_E2E_URAM_Q=0 -UHGTXR_E2E_LUTRAM_Q -DHGTXR_E2E_LUTRAM_Q=0 -UHGTXR_E2E_URAM_K -DHGTXR_E2E_URAM_K=0 -UHGTXR_E2E_LUTRAM_K -DHGTXR_E2E_LUTRAM_K=0 -UHGTXR_E2E_URAM_V -DHGTXR_E2E_URAM_V=0 -UHGTXR_E2E_LUTRAM_V -DHGTXR_E2E_LUTRAM_V=0 -UHGTXR_E2E_URAM_ATTN -DHGTXR_E2E_URAM_ATTN=0 -UHGTXR_E2E_LUTRAM_ATTN -DHGTXR_E2E_LUTRAM_ATTN=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_CORE_FABRIC_TAIL_LANES -DHGTXR_E2E_CORE_FABRIC_TAIL_LANES=2 -UHGTXR_E2E_CORE_LANE_CT_SWITCH -DHGTXR_E2E_CORE_LANE_CT_SWITCH=1 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=8 -UHGTXR_E2E_PATCH_PAR -DHGTXR_E2E_PATCH_PAR=32 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -UHGTXR_E2E_MLP_TOKEN_PAR -DHGTXR_E2E_MLP_TOKEN_PAR=4 -UHGTXR_E2E_DENSE_TOKEN_PAR -DHGTXR_E2E_DENSE_TOKEN_PAR=4 -UHGTXR_E2E_ATTN_QUERY_PAR -DHGTXR_E2E_ATTN_QUERY_PAR=8 -DHGTXR_E2E_MLP_FUSED_W2=1 -DHGTXR_E2E_MLP_FUSED_W2_BANKED_ACC=1 -DHGTXR_E2E_MLP_FUSED_W2_HP_PAR=16 -DHGTXR_E2E_MLP_W1_C_PAR=2 -DHGTXR_E2E_MLP_W2_HGROUP_BANK_CACHE=1 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_GELU_ROM_PARTITION=1 -DHGTXR_E2E_SOFTMAX_EXP_ROM_PARTITION=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -UHGTXR_E2E_TEST_RUNTIME_MODES -DHGTXR_E2E_TEST_RUNTIME_MODES=0 -DHGTXR_E2E_FORCE_RUNTIME_MODE=1 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar8_mlpfusebank_hpar8_acc24_w2hgroup_mtok4_dtok4_aq8_qkvbram_tail2_exppart_w1c2_attntok2_attnbram_nocache_patchpar64_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only)
    PROJECT="hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq8_hp8_acc24_patch64_w2hgroup_qkvbram_tail2_exppart_w1c2_300_mem16_search_only_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_ACC_W -DHGTXR_ACC_W=24 -UHGTXR_ACC_I -DHGTXR_ACC_I=10 -UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_URAM_Q -DHGTXR_E2E_URAM_Q=0 -UHGTXR_E2E_LUTRAM_Q -DHGTXR_E2E_LUTRAM_Q=0 -UHGTXR_E2E_URAM_K -DHGTXR_E2E_URAM_K=0 -UHGTXR_E2E_LUTRAM_K -DHGTXR_E2E_LUTRAM_K=0 -UHGTXR_E2E_URAM_V -DHGTXR_E2E_URAM_V=0 -UHGTXR_E2E_LUTRAM_V -DHGTXR_E2E_LUTRAM_V=0 -UHGTXR_E2E_URAM_ATTN -DHGTXR_E2E_URAM_ATTN=0 -UHGTXR_E2E_LUTRAM_ATTN -DHGTXR_E2E_LUTRAM_ATTN=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_CORE_FABRIC_TAIL_LANES -DHGTXR_E2E_CORE_FABRIC_TAIL_LANES=2 -UHGTXR_E2E_CORE_LANE_CT_SWITCH -DHGTXR_E2E_CORE_LANE_CT_SWITCH=1 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=8 -UHGTXR_E2E_PATCH_PAR -DHGTXR_E2E_PATCH_PAR=64 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -UHGTXR_E2E_MLP_TOKEN_PAR -DHGTXR_E2E_MLP_TOKEN_PAR=4 -UHGTXR_E2E_DENSE_TOKEN_PAR -DHGTXR_E2E_DENSE_TOKEN_PAR=4 -UHGTXR_E2E_ATTN_QUERY_PAR -DHGTXR_E2E_ATTN_QUERY_PAR=8 -DHGTXR_E2E_MLP_FUSED_W2=1 -DHGTXR_E2E_MLP_FUSED_W2_BANKED_ACC=1 -DHGTXR_E2E_MLP_FUSED_W2_HP_PAR=8 -DHGTXR_E2E_MLP_W1_C_PAR=2 -DHGTXR_E2E_MLP_W2_HGROUP_BANK_CACHE=1 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_SOFTMAX_EXP_ROM_PARTITION=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -UHGTXR_E2E_TEST_RUNTIME_MODES -DHGTXR_E2E_TEST_RUNTIME_MODES=0 -DHGTXR_E2E_FORCE_RUNTIME_MODE=0 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar8_mlpfusebank_hpar8_acc24_tokbank4_w2hgroup_mtok4_dtok4_aq8_qkvbram_tail2_exppart_w1c2_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only)
    PROJECT="hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq8_hp8_acc24_tokbank4_w2hgroup_qkvbram_tail2_exppart_w1c2_300_mem16_search_only_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_ACC_W -DHGTXR_ACC_W=24 -UHGTXR_ACC_I -DHGTXR_ACC_I=10 -DHGTXR_E2E_TOKEN_BANK_PAR=4 -UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_URAM_Q -DHGTXR_E2E_URAM_Q=0 -UHGTXR_E2E_LUTRAM_Q -DHGTXR_E2E_LUTRAM_Q=0 -UHGTXR_E2E_URAM_K -DHGTXR_E2E_URAM_K=0 -UHGTXR_E2E_LUTRAM_K -DHGTXR_E2E_LUTRAM_K=0 -UHGTXR_E2E_URAM_V -DHGTXR_E2E_URAM_V=0 -UHGTXR_E2E_LUTRAM_V -DHGTXR_E2E_LUTRAM_V=0 -UHGTXR_E2E_URAM_ATTN -DHGTXR_E2E_URAM_ATTN=0 -UHGTXR_E2E_LUTRAM_ATTN -DHGTXR_E2E_LUTRAM_ATTN=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_CORE_FABRIC_TAIL_LANES -DHGTXR_E2E_CORE_FABRIC_TAIL_LANES=2 -UHGTXR_E2E_CORE_LANE_CT_SWITCH -DHGTXR_E2E_CORE_LANE_CT_SWITCH=1 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=8 -UHGTXR_E2E_PATCH_PAR -DHGTXR_E2E_PATCH_PAR=32 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -UHGTXR_E2E_MLP_TOKEN_PAR -DHGTXR_E2E_MLP_TOKEN_PAR=4 -UHGTXR_E2E_DENSE_TOKEN_PAR -DHGTXR_E2E_DENSE_TOKEN_PAR=4 -UHGTXR_E2E_ATTN_QUERY_PAR -DHGTXR_E2E_ATTN_QUERY_PAR=8 -DHGTXR_E2E_MLP_FUSED_W2=1 -DHGTXR_E2E_MLP_FUSED_W2_BANKED_ACC=1 -DHGTXR_E2E_MLP_FUSED_W2_HP_PAR=8 -DHGTXR_E2E_MLP_W1_C_PAR=2 -DHGTXR_E2E_MLP_W2_HGROUP_BANK_CACHE=1 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_SOFTMAX_EXP_ROM_PARTITION=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -UHGTXR_E2E_TEST_RUNTIME_MODES -DHGTXR_E2E_TEST_RUNTIME_MODES=0 -DHGTXR_E2E_FORCE_RUNTIME_MODE=0 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar8_mlpfusebank_hpar8_acc24_nowide_w2hgroup_mtok4_dtok4_aq8_qkvbram_tail2_exppart_w1c2_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only)
    PROJECT="hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq8_hp8_acc24_nowide_w2hgroup_qkvbram_tail2_exppart_w1c2_300_mem16_search_only_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_ACC_W -DHGTXR_ACC_W=24 -UHGTXR_ACC_I -DHGTXR_ACC_I=10 -UHGTXR_E2E_FORCE_WIDE_DSP_MUL -DHGTXR_E2E_FORCE_WIDE_DSP_MUL=0 -UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_URAM_Q -DHGTXR_E2E_URAM_Q=0 -UHGTXR_E2E_LUTRAM_Q -DHGTXR_E2E_LUTRAM_Q=0 -UHGTXR_E2E_URAM_K -DHGTXR_E2E_URAM_K=0 -UHGTXR_E2E_LUTRAM_K -DHGTXR_E2E_LUTRAM_K=0 -UHGTXR_E2E_URAM_V -DHGTXR_E2E_URAM_V=0 -UHGTXR_E2E_LUTRAM_V -DHGTXR_E2E_LUTRAM_V=0 -UHGTXR_E2E_URAM_ATTN -DHGTXR_E2E_URAM_ATTN=0 -UHGTXR_E2E_LUTRAM_ATTN -DHGTXR_E2E_LUTRAM_ATTN=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_CORE_FABRIC_TAIL_LANES -DHGTXR_E2E_CORE_FABRIC_TAIL_LANES=2 -UHGTXR_E2E_CORE_LANE_CT_SWITCH -DHGTXR_E2E_CORE_LANE_CT_SWITCH=1 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=8 -UHGTXR_E2E_PATCH_PAR -DHGTXR_E2E_PATCH_PAR=32 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -UHGTXR_E2E_MLP_TOKEN_PAR -DHGTXR_E2E_MLP_TOKEN_PAR=4 -UHGTXR_E2E_DENSE_TOKEN_PAR -DHGTXR_E2E_DENSE_TOKEN_PAR=4 -UHGTXR_E2E_ATTN_QUERY_PAR -DHGTXR_E2E_ATTN_QUERY_PAR=8 -DHGTXR_E2E_MLP_FUSED_W2=1 -DHGTXR_E2E_MLP_FUSED_W2_BANKED_ACC=1 -DHGTXR_E2E_MLP_FUSED_W2_HP_PAR=8 -DHGTXR_E2E_MLP_W1_C_PAR=2 -DHGTXR_E2E_MLP_W2_HGROUP_BANK_CACHE=1 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_SOFTMAX_EXP_ROM_PARTITION=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -UHGTXR_E2E_TEST_RUNTIME_MODES -DHGTXR_E2E_TEST_RUNTIME_MODES=0 -DHGTXR_E2E_FORCE_RUNTIME_MODE=0 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar8_mlpfusebank_hpar8_w2hgroup_mtok4_dtok4_aq8_qkvbram_tail2_exppart_w1c2_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_track_only)
    PROJECT="hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq8_hp8_w2hgroup_qkvbram_tail2_exppart_w1c2_300_mem16_track_only_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_URAM_Q -DHGTXR_E2E_URAM_Q=0 -UHGTXR_E2E_LUTRAM_Q -DHGTXR_E2E_LUTRAM_Q=0 -UHGTXR_E2E_URAM_K -DHGTXR_E2E_URAM_K=0 -UHGTXR_E2E_LUTRAM_K -DHGTXR_E2E_LUTRAM_K=0 -UHGTXR_E2E_URAM_V -DHGTXR_E2E_URAM_V=0 -UHGTXR_E2E_LUTRAM_V -DHGTXR_E2E_LUTRAM_V=0 -UHGTXR_E2E_URAM_ATTN -DHGTXR_E2E_URAM_ATTN=0 -UHGTXR_E2E_LUTRAM_ATTN -DHGTXR_E2E_LUTRAM_ATTN=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_CORE_FABRIC_TAIL_LANES -DHGTXR_E2E_CORE_FABRIC_TAIL_LANES=2 -UHGTXR_E2E_CORE_LANE_CT_SWITCH -DHGTXR_E2E_CORE_LANE_CT_SWITCH=1 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=8 -UHGTXR_E2E_PATCH_PAR -DHGTXR_E2E_PATCH_PAR=32 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -UHGTXR_E2E_MLP_TOKEN_PAR -DHGTXR_E2E_MLP_TOKEN_PAR=4 -UHGTXR_E2E_DENSE_TOKEN_PAR -DHGTXR_E2E_DENSE_TOKEN_PAR=4 -UHGTXR_E2E_ATTN_QUERY_PAR -DHGTXR_E2E_ATTN_QUERY_PAR=8 -DHGTXR_E2E_MLP_FUSED_W2=1 -DHGTXR_E2E_MLP_FUSED_W2_BANKED_ACC=1 -DHGTXR_E2E_MLP_FUSED_W2_HP_PAR=8 -DHGTXR_E2E_MLP_W1_C_PAR=2 -DHGTXR_E2E_MLP_W2_HGROUP_BANK_CACHE=1 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_SOFTMAX_EXP_ROM_PARTITION=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -UHGTXR_E2E_TEST_RUNTIME_MODES -DHGTXR_E2E_TEST_RUNTIME_MODES=0 -DHGTXR_E2E_FORCE_RUNTIME_MODE=1 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar8_mlpfusebank_hpar8_skiph_w2hgroup_mtok4_dtok4_aq8_qkvbram_tail2_exppart_w1c2_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only)
    PROJECT="hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq8_hp8_skiph_w2hgroup_qkvbram_tail2_exppart_w1c2_300_mem16_search_only_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_URAM_Q -DHGTXR_E2E_URAM_Q=0 -UHGTXR_E2E_LUTRAM_Q -DHGTXR_E2E_LUTRAM_Q=0 -UHGTXR_E2E_URAM_K -DHGTXR_E2E_URAM_K=0 -UHGTXR_E2E_LUTRAM_K -DHGTXR_E2E_LUTRAM_K=0 -UHGTXR_E2E_URAM_V -DHGTXR_E2E_URAM_V=0 -UHGTXR_E2E_LUTRAM_V -DHGTXR_E2E_LUTRAM_V=0 -UHGTXR_E2E_URAM_ATTN -DHGTXR_E2E_URAM_ATTN=0 -UHGTXR_E2E_LUTRAM_ATTN -DHGTXR_E2E_LUTRAM_ATTN=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_CORE_FABRIC_TAIL_LANES -DHGTXR_E2E_CORE_FABRIC_TAIL_LANES=2 -UHGTXR_E2E_CORE_LANE_CT_SWITCH -DHGTXR_E2E_CORE_LANE_CT_SWITCH=1 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=8 -UHGTXR_E2E_PATCH_PAR -DHGTXR_E2E_PATCH_PAR=32 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -UHGTXR_E2E_MLP_TOKEN_PAR -DHGTXR_E2E_MLP_TOKEN_PAR=4 -UHGTXR_E2E_DENSE_TOKEN_PAR -DHGTXR_E2E_DENSE_TOKEN_PAR=4 -UHGTXR_E2E_ATTN_QUERY_PAR -DHGTXR_E2E_ATTN_QUERY_PAR=8 -DHGTXR_E2E_MLP_FUSED_W2=1 -DHGTXR_E2E_MLP_FUSED_W2_BANKED_ACC=1 -DHGTXR_E2E_MLP_FUSED_W2_SKIP_HIDDEN_STORE=1 -DHGTXR_E2E_MLP_FUSED_W2_HP_PAR=8 -DHGTXR_E2E_MLP_W1_C_PAR=2 -DHGTXR_E2E_MLP_W2_HGROUP_BANK_CACHE=1 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_SOFTMAX_EXP_ROM_PARTITION=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -UHGTXR_E2E_TEST_RUNTIME_MODES -DHGTXR_E2E_TEST_RUNTIME_MODES=0 -DHGTXR_E2E_FORCE_RUNTIME_MODE=0 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar8_mlpfusebank_hpar8_lncache_w2hgroup_mtok4_dtok4_aq8_qkvbram_tail2_exppart_w1c2_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only)
    PROJECT="hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq8_hp8_lncache_w2hgroup_qkvbram_tail2_exppart_w1c2_300_mem16_search_only_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_URAM_Q -DHGTXR_E2E_URAM_Q=0 -UHGTXR_E2E_LUTRAM_Q -DHGTXR_E2E_LUTRAM_Q=0 -UHGTXR_E2E_URAM_K -DHGTXR_E2E_URAM_K=0 -UHGTXR_E2E_LUTRAM_K -DHGTXR_E2E_LUTRAM_K=0 -UHGTXR_E2E_URAM_V -DHGTXR_E2E_URAM_V=0 -UHGTXR_E2E_LUTRAM_V -DHGTXR_E2E_LUTRAM_V=0 -UHGTXR_E2E_URAM_ATTN -DHGTXR_E2E_URAM_ATTN=0 -UHGTXR_E2E_LUTRAM_ATTN -DHGTXR_E2E_LUTRAM_ATTN=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_CORE_FABRIC_TAIL_LANES -DHGTXR_E2E_CORE_FABRIC_TAIL_LANES=2 -UHGTXR_E2E_CORE_LANE_CT_SWITCH -DHGTXR_E2E_CORE_LANE_CT_SWITCH=1 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=1 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=8 -UHGTXR_E2E_PATCH_PAR -DHGTXR_E2E_PATCH_PAR=32 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -UHGTXR_E2E_MLP_TOKEN_PAR -DHGTXR_E2E_MLP_TOKEN_PAR=4 -UHGTXR_E2E_DENSE_TOKEN_PAR -DHGTXR_E2E_DENSE_TOKEN_PAR=4 -UHGTXR_E2E_ATTN_QUERY_PAR -DHGTXR_E2E_ATTN_QUERY_PAR=8 -DHGTXR_E2E_MLP_FUSED_W2=1 -DHGTXR_E2E_MLP_FUSED_W2_BANKED_ACC=1 -DHGTXR_E2E_MLP_FUSED_W2_HP_PAR=8 -DHGTXR_E2E_MLP_W1_C_PAR=2 -DHGTXR_E2E_MLP_W2_HGROUP_BANK_CACHE=1 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_SOFTMAX_EXP_ROM_PARTITION=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -UHGTXR_E2E_TEST_RUNTIME_MODES -DHGTXR_E2E_TEST_RUNTIME_MODES=0 -DHGTXR_E2E_FORCE_RUNTIME_MODE=0 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar8_mlpfusebank_hpar8_w2fl2_w2hgroup_mtok4_dtok4_aq8_qkvbram_tail2_exppart_w1c2_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only)
    PROJECT="hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq8_hp8_w2fl2_w2hgroup_qkvbram_tail2_exppart_w1c2_300_mem16_search_only_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_URAM_Q -DHGTXR_E2E_URAM_Q=0 -UHGTXR_E2E_LUTRAM_Q -DHGTXR_E2E_LUTRAM_Q=0 -UHGTXR_E2E_URAM_K -DHGTXR_E2E_URAM_K=0 -UHGTXR_E2E_LUTRAM_K -DHGTXR_E2E_LUTRAM_K=0 -UHGTXR_E2E_URAM_V -DHGTXR_E2E_URAM_V=0 -UHGTXR_E2E_LUTRAM_V -DHGTXR_E2E_LUTRAM_V=0 -UHGTXR_E2E_URAM_ATTN -DHGTXR_E2E_URAM_ATTN=0 -UHGTXR_E2E_LUTRAM_ATTN -DHGTXR_E2E_LUTRAM_ATTN=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_CORE_FABRIC_TAIL_LANES -DHGTXR_E2E_CORE_FABRIC_TAIL_LANES=2 -UHGTXR_E2E_CORE_LANE_CT_SWITCH -DHGTXR_E2E_CORE_LANE_CT_SWITCH=1 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=8 -UHGTXR_E2E_PATCH_PAR -DHGTXR_E2E_PATCH_PAR=32 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -UHGTXR_E2E_MLP_TOKEN_PAR -DHGTXR_E2E_MLP_TOKEN_PAR=4 -UHGTXR_E2E_DENSE_TOKEN_PAR -DHGTXR_E2E_DENSE_TOKEN_PAR=4 -UHGTXR_E2E_ATTN_QUERY_PAR -DHGTXR_E2E_ATTN_QUERY_PAR=8 -DHGTXR_E2E_MLP_FUSED_W2=1 -DHGTXR_E2E_MLP_FUSED_W2_BANKED_ACC=1 -DHGTXR_E2E_MLP_FUSED_W2_HP_PAR=8 -DHGTXR_E2E_MLP_W2_FABRIC_HP_LANES=2 -DHGTXR_E2E_MLP_W1_C_PAR=2 -DHGTXR_E2E_MLP_W2_HGROUP_BANK_CACHE=1 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_SOFTMAX_EXP_ROM_PARTITION=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -UHGTXR_E2E_TEST_RUNTIME_MODES -DHGTXR_E2E_TEST_RUNTIME_MODES=0 -UHGTXR_E2E_FORCE_RUNTIME_MODE=0 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar4_w2hgroup_mtok4_dtok4_aq6_qkvbram_tail2_exppart_w1c2_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only)
    PROJECT="hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq6_hp4_w2hgroup_qkvbram_tail2_exppart_w1c2_300_mem16_search_only_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_URAM_Q -DHGTXR_E2E_URAM_Q=0 -UHGTXR_E2E_LUTRAM_Q -DHGTXR_E2E_LUTRAM_Q=0 -UHGTXR_E2E_URAM_K -DHGTXR_E2E_URAM_K=0 -UHGTXR_E2E_LUTRAM_K -DHGTXR_E2E_LUTRAM_K=0 -UHGTXR_E2E_URAM_V -DHGTXR_E2E_URAM_V=0 -UHGTXR_E2E_LUTRAM_V -DHGTXR_E2E_LUTRAM_V=0 -UHGTXR_E2E_URAM_ATTN -DHGTXR_E2E_URAM_ATTN=0 -UHGTXR_E2E_LUTRAM_ATTN -DHGTXR_E2E_LUTRAM_ATTN=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_CORE_FABRIC_TAIL_LANES -DHGTXR_E2E_CORE_FABRIC_TAIL_LANES=2 -UHGTXR_E2E_CORE_LANE_CT_SWITCH -DHGTXR_E2E_CORE_LANE_CT_SWITCH=1 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=4 -UHGTXR_E2E_PATCH_PAR -DHGTXR_E2E_PATCH_PAR=32 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -UHGTXR_E2E_MLP_TOKEN_PAR -DHGTXR_E2E_MLP_TOKEN_PAR=4 -UHGTXR_E2E_DENSE_TOKEN_PAR -DHGTXR_E2E_DENSE_TOKEN_PAR=4 -UHGTXR_E2E_ATTN_QUERY_PAR -DHGTXR_E2E_ATTN_QUERY_PAR=6 -DHGTXR_E2E_MLP_FUSED_W2=1 -DHGTXR_E2E_MLP_FUSED_W2_BANKED_ACC=1 -DHGTXR_E2E_MLP_FUSED_W2_HP_PAR=4 -DHGTXR_E2E_MLP_W1_C_PAR=2 -DHGTXR_E2E_MLP_W2_HGROUP_BANK_CACHE=1 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_SOFTMAX_EXP_ROM_PARTITION=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -UHGTXR_E2E_TEST_RUNTIME_MODES -DHGTXR_E2E_TEST_RUNTIME_MODES=0 -DHGTXR_E2E_FORCE_RUNTIME_MODE=0 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar4_w2hgroup_mtok4_dtok4_aq8_qkvbram_tail8_exppart_w1c2_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only)
    PROJECT="hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq8_hp4_w2hgroup_qkvbram_tail8_exppart_w1c2_300_mem16_search_only_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_URAM_Q -DHGTXR_E2E_URAM_Q=0 -UHGTXR_E2E_LUTRAM_Q -DHGTXR_E2E_LUTRAM_Q=0 -UHGTXR_E2E_URAM_K -DHGTXR_E2E_URAM_K=0 -UHGTXR_E2E_LUTRAM_K -DHGTXR_E2E_LUTRAM_K=0 -UHGTXR_E2E_URAM_V -DHGTXR_E2E_URAM_V=0 -UHGTXR_E2E_LUTRAM_V -DHGTXR_E2E_LUTRAM_V=0 -UHGTXR_E2E_URAM_ATTN -DHGTXR_E2E_URAM_ATTN=0 -UHGTXR_E2E_LUTRAM_ATTN -DHGTXR_E2E_LUTRAM_ATTN=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_CORE_FABRIC_TAIL_LANES -DHGTXR_E2E_CORE_FABRIC_TAIL_LANES=8 -UHGTXR_E2E_CORE_LANE_CT_SWITCH -DHGTXR_E2E_CORE_LANE_CT_SWITCH=1 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=4 -UHGTXR_E2E_PATCH_PAR -DHGTXR_E2E_PATCH_PAR=32 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -UHGTXR_E2E_MLP_TOKEN_PAR -DHGTXR_E2E_MLP_TOKEN_PAR=4 -UHGTXR_E2E_DENSE_TOKEN_PAR -DHGTXR_E2E_DENSE_TOKEN_PAR=4 -UHGTXR_E2E_ATTN_QUERY_PAR -DHGTXR_E2E_ATTN_QUERY_PAR=8 -DHGTXR_E2E_MLP_FUSED_W2=1 -DHGTXR_E2E_MLP_FUSED_W2_BANKED_ACC=1 -DHGTXR_E2E_MLP_FUSED_W2_HP_PAR=4 -DHGTXR_E2E_MLP_W1_C_PAR=2 -DHGTXR_E2E_MLP_W2_HGROUP_BANK_CACHE=1 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_SOFTMAX_EXP_ROM_PARTITION=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -UHGTXR_E2E_TEST_RUNTIME_MODES -DHGTXR_E2E_TEST_RUNTIME_MODES=0 -DHGTXR_E2E_FORCE_RUNTIME_MODE=0 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar4_w2hgroup_mtok4_dtok4_aq8_qkvbram_tail2_exppart_w1c4_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only)
    PROJECT="hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq8_hp4_w2hgroup_qkvbram_tail2_exppart_w1c4_300_mem16_search_only_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_URAM_Q -DHGTXR_E2E_URAM_Q=0 -UHGTXR_E2E_LUTRAM_Q -DHGTXR_E2E_LUTRAM_Q=0 -UHGTXR_E2E_URAM_K -DHGTXR_E2E_URAM_K=0 -UHGTXR_E2E_LUTRAM_K -DHGTXR_E2E_LUTRAM_K=0 -UHGTXR_E2E_URAM_V -DHGTXR_E2E_URAM_V=0 -UHGTXR_E2E_LUTRAM_V -DHGTXR_E2E_LUTRAM_V=0 -UHGTXR_E2E_URAM_ATTN -DHGTXR_E2E_URAM_ATTN=0 -UHGTXR_E2E_LUTRAM_ATTN -DHGTXR_E2E_LUTRAM_ATTN=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_CORE_FABRIC_TAIL_LANES -DHGTXR_E2E_CORE_FABRIC_TAIL_LANES=2 -UHGTXR_E2E_CORE_LANE_CT_SWITCH -DHGTXR_E2E_CORE_LANE_CT_SWITCH=1 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=4 -UHGTXR_E2E_PATCH_PAR -DHGTXR_E2E_PATCH_PAR=32 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -UHGTXR_E2E_MLP_TOKEN_PAR -DHGTXR_E2E_MLP_TOKEN_PAR=4 -UHGTXR_E2E_DENSE_TOKEN_PAR -DHGTXR_E2E_DENSE_TOKEN_PAR=4 -UHGTXR_E2E_ATTN_QUERY_PAR -DHGTXR_E2E_ATTN_QUERY_PAR=8 -DHGTXR_E2E_MLP_FUSED_W2=1 -DHGTXR_E2E_MLP_FUSED_W2_BANKED_ACC=1 -DHGTXR_E2E_MLP_FUSED_W2_HP_PAR=4 -DHGTXR_E2E_MLP_W1_C_PAR=4 -DHGTXR_E2E_MLP_W2_HGROUP_BANK_CACHE=1 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_SOFTMAX_EXP_ROM_PARTITION=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -UHGTXR_E2E_TEST_RUNTIME_MODES -DHGTXR_E2E_TEST_RUNTIME_MODES=0 -DHGTXR_E2E_FORCE_RUNTIME_MODE=0 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar4_w2hgroup_mtok4_dtok4_aq8_qkvbram_tail2_exppart_w1c4_w1bank16_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only)
    PROJECT="hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq8_hp4_w2hgroup_qkvbram_tail2_exppart_w1c4_w1b16_300_mem16_search_only_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_URAM_Q -DHGTXR_E2E_URAM_Q=0 -UHGTXR_E2E_LUTRAM_Q -DHGTXR_E2E_LUTRAM_Q=0 -UHGTXR_E2E_URAM_K -DHGTXR_E2E_URAM_K=0 -UHGTXR_E2E_LUTRAM_K -DHGTXR_E2E_LUTRAM_K=0 -UHGTXR_E2E_URAM_V -DHGTXR_E2E_URAM_V=0 -UHGTXR_E2E_LUTRAM_V -DHGTXR_E2E_LUTRAM_V=0 -UHGTXR_E2E_URAM_ATTN -DHGTXR_E2E_URAM_ATTN=0 -UHGTXR_E2E_LUTRAM_ATTN -DHGTXR_E2E_LUTRAM_ATTN=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_CORE_FABRIC_TAIL_LANES -DHGTXR_E2E_CORE_FABRIC_TAIL_LANES=2 -UHGTXR_E2E_CORE_LANE_CT_SWITCH -DHGTXR_E2E_CORE_LANE_CT_SWITCH=1 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=4 -UHGTXR_E2E_PATCH_PAR -DHGTXR_E2E_PATCH_PAR=32 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -UHGTXR_E2E_MLP_TOKEN_PAR -DHGTXR_E2E_MLP_TOKEN_PAR=4 -UHGTXR_E2E_DENSE_TOKEN_PAR -DHGTXR_E2E_DENSE_TOKEN_PAR=4 -UHGTXR_E2E_ATTN_QUERY_PAR -DHGTXR_E2E_ATTN_QUERY_PAR=8 -DHGTXR_E2E_MLP_FUSED_W2=1 -DHGTXR_E2E_MLP_FUSED_W2_BANKED_ACC=1 -DHGTXR_E2E_MLP_FUSED_W2_HP_PAR=4 -DHGTXR_E2E_MLP_W1_C_PAR=4 -DHGTXR_E2E_W1_WEIGHT_CACHE_BANKS=16 -DHGTXR_E2E_MLP_W2_HGROUP_BANK_CACHE=1 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_SOFTMAX_EXP_ROM_PARTITION=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -UHGTXR_E2E_TEST_RUNTIME_MODES -DHGTXR_E2E_TEST_RUNTIME_MODES=0 -DHGTXR_E2E_FORCE_RUNTIME_MODE=0 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar4_w2hgroup_mtok4_dtok4_aq4_qkvbram_tail4_exppart_w1c2_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only)
    PROJECT="hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq4_hp4_w2hgroup_qkvbram_tail4_exppart_w1c2_300_mem16_search_only_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_URAM_Q -DHGTXR_E2E_URAM_Q=0 -UHGTXR_E2E_LUTRAM_Q -DHGTXR_E2E_LUTRAM_Q=0 -UHGTXR_E2E_URAM_K -DHGTXR_E2E_URAM_K=0 -UHGTXR_E2E_LUTRAM_K -DHGTXR_E2E_LUTRAM_K=0 -UHGTXR_E2E_URAM_V -DHGTXR_E2E_URAM_V=0 -UHGTXR_E2E_LUTRAM_V -DHGTXR_E2E_LUTRAM_V=0 -UHGTXR_E2E_URAM_ATTN -DHGTXR_E2E_URAM_ATTN=0 -UHGTXR_E2E_LUTRAM_ATTN -DHGTXR_E2E_LUTRAM_ATTN=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_CORE_FABRIC_TAIL_LANES -DHGTXR_E2E_CORE_FABRIC_TAIL_LANES=4 -UHGTXR_E2E_CORE_LANE_CT_SWITCH -DHGTXR_E2E_CORE_LANE_CT_SWITCH=1 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=4 -UHGTXR_E2E_PATCH_PAR -DHGTXR_E2E_PATCH_PAR=32 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -UHGTXR_E2E_MLP_TOKEN_PAR -DHGTXR_E2E_MLP_TOKEN_PAR=4 -UHGTXR_E2E_DENSE_TOKEN_PAR -DHGTXR_E2E_DENSE_TOKEN_PAR=4 -UHGTXR_E2E_ATTN_QUERY_PAR -DHGTXR_E2E_ATTN_QUERY_PAR=4 -DHGTXR_E2E_MLP_FUSED_W2=1 -DHGTXR_E2E_MLP_FUSED_W2_BANKED_ACC=1 -DHGTXR_E2E_MLP_FUSED_W2_HP_PAR=4 -DHGTXR_E2E_MLP_W1_C_PAR=2 -DHGTXR_E2E_MLP_W2_HGROUP_BANK_CACHE=1 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_SOFTMAX_EXP_ROM_PARTITION=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -UHGTXR_E2E_TEST_RUNTIME_MODES -DHGTXR_E2E_TEST_RUNTIME_MODES=0 -DHGTXR_E2E_FORCE_RUNTIME_MODE=0 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_mtok8_dtok4_aq2_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only)
    PROJECT="hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_mtok8_dtok4_aq2_300_mem16_search_only_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_URAM_ATTN -DHGTXR_E2E_URAM_ATTN=0 -UHGTXR_E2E_LUTRAM_ATTN -DHGTXR_E2E_LUTRAM_ATTN=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=4 -UHGTXR_E2E_PATCH_PAR -DHGTXR_E2E_PATCH_PAR=32 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -UHGTXR_E2E_MLP_TOKEN_PAR -DHGTXR_E2E_MLP_TOKEN_PAR=8 -UHGTXR_E2E_DENSE_TOKEN_PAR -DHGTXR_E2E_DENSE_TOKEN_PAR=4 -UHGTXR_E2E_ATTN_QUERY_PAR -DHGTXR_E2E_ATTN_QUERY_PAR=2 -DHGTXR_E2E_MLP_FUSED_W2=1 -DHGTXR_E2E_MLP_FUSED_W2_BANKED_ACC=1 -DHGTXR_E2E_MLP_FUSED_W2_HP_PAR=2 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -UHGTXR_E2E_TEST_RUNTIME_MODES -DHGTXR_E2E_TEST_RUNTIME_MODES=0 -DHGTXR_E2E_FORCE_RUNTIME_MODE=0 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_mtok3_dtok4_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16)
    PROJECT="hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_mtok3_dtok4_300_mem16_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_URAM_ATTN -DHGTXR_E2E_URAM_ATTN=0 -UHGTXR_E2E_LUTRAM_ATTN -DHGTXR_E2E_LUTRAM_ATTN=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=4 -UHGTXR_E2E_PATCH_PAR -DHGTXR_E2E_PATCH_PAR=32 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -UHGTXR_E2E_MLP_TOKEN_PAR -DHGTXR_E2E_MLP_TOKEN_PAR=3 -UHGTXR_E2E_DENSE_TOKEN_PAR -DHGTXR_E2E_DENSE_TOKEN_PAR=4 -DHGTXR_E2E_MLP_FUSED_W2=1 -DHGTXR_E2E_MLP_FUSED_W2_BANKED_ACC=1 -DHGTXR_E2E_MLP_FUSED_W2_HP_PAR=2 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_mtok3_dtok4_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only)
    PROJECT="hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_mtok3_dtok4_300_mem16_search_only_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_URAM_ATTN -DHGTXR_E2E_URAM_ATTN=0 -UHGTXR_E2E_LUTRAM_ATTN -DHGTXR_E2E_LUTRAM_ATTN=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=4 -UHGTXR_E2E_PATCH_PAR -DHGTXR_E2E_PATCH_PAR=32 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -UHGTXR_E2E_MLP_TOKEN_PAR -DHGTXR_E2E_MLP_TOKEN_PAR=3 -UHGTXR_E2E_DENSE_TOKEN_PAR -DHGTXR_E2E_DENSE_TOKEN_PAR=4 -DHGTXR_E2E_MLP_FUSED_W2=1 -DHGTXR_E2E_MLP_FUSED_W2_BANKED_ACC=1 -DHGTXR_E2E_MLP_FUSED_W2_HP_PAR=2 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -UHGTXR_E2E_TEST_RUNTIME_MODES -DHGTXR_E2E_TEST_RUNTIME_MODES=0 -DHGTXR_E2E_FORCE_RUNTIME_MODE=0 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_mtok3_dtok4_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_track_only)
    PROJECT="hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_mtok3_dtok4_300_mem16_track_only_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_URAM_ATTN -DHGTXR_E2E_URAM_ATTN=0 -UHGTXR_E2E_LUTRAM_ATTN -DHGTXR_E2E_LUTRAM_ATTN=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=4 -UHGTXR_E2E_PATCH_PAR -DHGTXR_E2E_PATCH_PAR=32 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -UHGTXR_E2E_MLP_TOKEN_PAR -DHGTXR_E2E_MLP_TOKEN_PAR=3 -UHGTXR_E2E_DENSE_TOKEN_PAR -DHGTXR_E2E_DENSE_TOKEN_PAR=4 -DHGTXR_E2E_MLP_FUSED_W2=1 -DHGTXR_E2E_MLP_FUSED_W2_BANKED_ACC=1 -DHGTXR_E2E_MLP_FUSED_W2_HP_PAR=2 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -UHGTXR_E2E_TEST_RUNTIME_MODES -DHGTXR_E2E_TEST_RUNTIME_MODES=0 -DHGTXR_E2E_FORCE_RUNTIME_MODE=1 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_mtok2_dtok3_attntok2_attnbram_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16)
    PROJECT="hgtxr_e2e_axis_par32_rt_attntok2_attnbram_dtok3_300_mem16_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_URAM_ATTN -DHGTXR_E2E_URAM_ATTN=0 -UHGTXR_E2E_LUTRAM_ATTN -DHGTXR_E2E_LUTRAM_ATTN=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=4 -DHGTXR_E2E_PATCH_PAR=16 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -UHGTXR_E2E_MLP_TOKEN_PAR -DHGTXR_E2E_MLP_TOKEN_PAR=2 -UHGTXR_E2E_DENSE_TOKEN_PAR -DHGTXR_E2E_DENSE_TOKEN_PAR=3 -DHGTXR_E2E_MLP_FUSED_W2=1 -DHGTXR_E2E_MLP_FUSED_W2_BANKED_ACC=1 -DHGTXR_E2E_MLP_FUSED_W2_HP_PAR=2 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_mtok2_dtok3_attntok2_attnbram_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only)
    PROJECT="hgtxr_e2e_axis_par32_rt_attntok2_attnbram_dtok3_300_mem16_search_only_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_URAM_ATTN -DHGTXR_E2E_URAM_ATTN=0 -UHGTXR_E2E_LUTRAM_ATTN -DHGTXR_E2E_LUTRAM_ATTN=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=4 -DHGTXR_E2E_PATCH_PAR=16 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -UHGTXR_E2E_MLP_TOKEN_PAR -DHGTXR_E2E_MLP_TOKEN_PAR=2 -UHGTXR_E2E_DENSE_TOKEN_PAR -DHGTXR_E2E_DENSE_TOKEN_PAR=3 -DHGTXR_E2E_MLP_FUSED_W2=1 -DHGTXR_E2E_MLP_FUSED_W2_BANKED_ACC=1 -DHGTXR_E2E_MLP_FUSED_W2_HP_PAR=2 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -UHGTXR_E2E_TEST_RUNTIME_MODES -DHGTXR_E2E_TEST_RUNTIME_MODES=0 -DHGTXR_E2E_FORCE_RUNTIME_MODE=0 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_mtok2_dtok3_attntok2_attnbram_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_track_only)
    PROJECT="hgtxr_e2e_axis_par32_rt_attntok2_attnbram_dtok3_300_mem16_track_only_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_URAM_ATTN -DHGTXR_E2E_URAM_ATTN=0 -UHGTXR_E2E_LUTRAM_ATTN -DHGTXR_E2E_LUTRAM_ATTN=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=4 -DHGTXR_E2E_PATCH_PAR=16 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -UHGTXR_E2E_MLP_TOKEN_PAR -DHGTXR_E2E_MLP_TOKEN_PAR=2 -UHGTXR_E2E_DENSE_TOKEN_PAR -DHGTXR_E2E_DENSE_TOKEN_PAR=3 -DHGTXR_E2E_MLP_FUSED_W2=1 -DHGTXR_E2E_MLP_FUSED_W2_BANKED_ACC=1 -DHGTXR_E2E_MLP_FUSED_W2_HP_PAR=2 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -UHGTXR_E2E_TEST_RUNTIME_MODES -DHGTXR_E2E_TEST_RUNTIME_MODES=0 -DHGTXR_E2E_FORCE_RUNTIME_MODE=1 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_skiphidden_tok2_attntok2_attnbram_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16)
    PROJECT="hgtxr_e2e_axis_par32_rt_attntok2_attnbram_skiphid_300_mem16_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_URAM_ATTN -DHGTXR_E2E_URAM_ATTN=0 -UHGTXR_E2E_LUTRAM_ATTN -DHGTXR_E2E_LUTRAM_ATTN=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=4 -DHGTXR_E2E_PATCH_PAR=16 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -DHGTXR_E2E_MLP_TOKEN_PAR=2 -DHGTXR_E2E_DENSE_TOKEN_PAR=2 -DHGTXR_E2E_MLP_FUSED_W2=1 -DHGTXR_E2E_MLP_FUSED_W2_BANKED_ACC=1 -DHGTXR_E2E_MLP_FUSED_W2_HP_PAR=2 -DHGTXR_E2E_MLP_FUSED_W2_SKIP_HIDDEN_STORE=1 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_skiphidden_tok2_attntok2_attnbram_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only)
    PROJECT="hgtxr_e2e_axis_par32_rt_attntok2_attnbram_skiphid_300_mem16_search_only_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_URAM_ATTN -DHGTXR_E2E_URAM_ATTN=0 -UHGTXR_E2E_LUTRAM_ATTN -DHGTXR_E2E_LUTRAM_ATTN=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=4 -DHGTXR_E2E_PATCH_PAR=16 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -DHGTXR_E2E_MLP_TOKEN_PAR=2 -DHGTXR_E2E_DENSE_TOKEN_PAR=2 -DHGTXR_E2E_MLP_FUSED_W2=1 -DHGTXR_E2E_MLP_FUSED_W2_BANKED_ACC=1 -DHGTXR_E2E_MLP_FUSED_W2_HP_PAR=2 -DHGTXR_E2E_MLP_FUSED_W2_SKIP_HIDDEN_STORE=1 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -UHGTXR_E2E_TEST_RUNTIME_MODES -DHGTXR_E2E_TEST_RUNTIME_MODES=0 -DHGTXR_E2E_FORCE_RUNTIME_MODE=0 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_skiphidden_tok2_attntok2_attnbram_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_track_only)
    PROJECT="hgtxr_e2e_axis_par32_rt_attntok2_attnbram_skiphid_300_mem16_track_only_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_URAM_ATTN -DHGTXR_E2E_URAM_ATTN=0 -UHGTXR_E2E_LUTRAM_ATTN -DHGTXR_E2E_LUTRAM_ATTN=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=4 -DHGTXR_E2E_PATCH_PAR=16 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -DHGTXR_E2E_MLP_TOKEN_PAR=2 -DHGTXR_E2E_DENSE_TOKEN_PAR=2 -DHGTXR_E2E_MLP_FUSED_W2=1 -DHGTXR_E2E_MLP_FUSED_W2_BANKED_ACC=1 -DHGTXR_E2E_MLP_FUSED_W2_HP_PAR=2 -DHGTXR_E2E_MLP_FUSED_W2_SKIP_HIDDEN_STORE=1 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -UHGTXR_E2E_TEST_RUNTIME_MODES -DHGTXR_E2E_TEST_RUNTIME_MODES=0 -DHGTXR_E2E_FORCE_RUNTIME_MODE=1 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar4_tok2_attntok2_attnbram_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16)
    PROJECT="hgtxr_e2e_axis_par32_rt_attntok2_attnbram_hpar4_300_mem16_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_URAM_ATTN -DHGTXR_E2E_URAM_ATTN=0 -UHGTXR_E2E_LUTRAM_ATTN -DHGTXR_E2E_LUTRAM_ATTN=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=4 -DHGTXR_E2E_PATCH_PAR=16 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -DHGTXR_E2E_MLP_TOKEN_PAR=2 -DHGTXR_E2E_DENSE_TOKEN_PAR=2 -DHGTXR_E2E_MLP_FUSED_W2=1 -DHGTXR_E2E_MLP_FUSED_W2_BANKED_ACC=1 -DHGTXR_E2E_MLP_FUSED_W2_HP_PAR=4 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar4_tok2_attntok2_attnbram_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only)
    PROJECT="hgtxr_e2e_axis_par32_rt_attntok2_attnbram_hpar4_300_mem16_search_only_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_URAM_ATTN -DHGTXR_E2E_URAM_ATTN=0 -UHGTXR_E2E_LUTRAM_ATTN -DHGTXR_E2E_LUTRAM_ATTN=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=4 -DHGTXR_E2E_PATCH_PAR=16 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -DHGTXR_E2E_MLP_TOKEN_PAR=2 -DHGTXR_E2E_DENSE_TOKEN_PAR=2 -DHGTXR_E2E_MLP_FUSED_W2=1 -DHGTXR_E2E_MLP_FUSED_W2_BANKED_ACC=1 -DHGTXR_E2E_MLP_FUSED_W2_HP_PAR=4 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -UHGTXR_E2E_TEST_RUNTIME_MODES -DHGTXR_E2E_TEST_RUNTIME_MODES=0 -DHGTXR_E2E_FORCE_RUNTIME_MODE=0 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar4_tok2_attntok2_attnbram_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_track_only)
    PROJECT="hgtxr_e2e_axis_par32_rt_attntok2_attnbram_hpar4_300_mem16_track_only_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_URAM_ATTN -DHGTXR_E2E_URAM_ATTN=0 -UHGTXR_E2E_LUTRAM_ATTN -DHGTXR_E2E_LUTRAM_ATTN=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=4 -DHGTXR_E2E_PATCH_PAR=16 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -DHGTXR_E2E_MLP_TOKEN_PAR=2 -DHGTXR_E2E_DENSE_TOKEN_PAR=2 -DHGTXR_E2E_MLP_FUSED_W2=1 -DHGTXR_E2E_MLP_FUSED_W2_BANKED_ACC=1 -DHGTXR_E2E_MLP_FUSED_W2_HP_PAR=4 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -UHGTXR_E2E_TEST_RUNTIME_MODES -DHGTXR_E2E_TEST_RUNTIME_MODES=0 -DHGTXR_E2E_FORCE_RUNTIME_MODE=1 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar4_w2bank8_tok2_attntok2_attnbram_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16)
    PROJECT="hgtxr_e2e_axis_par32_rt_attntok2_attnbram_hpar4_w2b8_300_mem16_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_URAM_ATTN -DHGTXR_E2E_URAM_ATTN=0 -UHGTXR_E2E_LUTRAM_ATTN -DHGTXR_E2E_LUTRAM_ATTN=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=4 -DHGTXR_E2E_PATCH_PAR=16 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -DHGTXR_E2E_MLP_TOKEN_PAR=2 -DHGTXR_E2E_DENSE_TOKEN_PAR=2 -DHGTXR_E2E_MLP_FUSED_W2=1 -DHGTXR_E2E_MLP_FUSED_W2_BANKED_ACC=1 -DHGTXR_E2E_MLP_FUSED_W2_HP_PAR=4 -DHGTXR_E2E_W2_WEIGHT_CACHE_BANKS=8 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar4_w2bank8_tok2_attntok2_attnbram_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only)
    PROJECT="hgtxr_e2e_axis_par32_rt_attntok2_attnbram_hpar4_w2b8_300_mem16_search_only_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_URAM_ATTN -DHGTXR_E2E_URAM_ATTN=0 -UHGTXR_E2E_LUTRAM_ATTN -DHGTXR_E2E_LUTRAM_ATTN=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=4 -DHGTXR_E2E_PATCH_PAR=16 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -DHGTXR_E2E_MLP_TOKEN_PAR=2 -DHGTXR_E2E_DENSE_TOKEN_PAR=2 -DHGTXR_E2E_MLP_FUSED_W2=1 -DHGTXR_E2E_MLP_FUSED_W2_BANKED_ACC=1 -DHGTXR_E2E_MLP_FUSED_W2_HP_PAR=4 -DHGTXR_E2E_W2_WEIGHT_CACHE_BANKS=8 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -UHGTXR_E2E_TEST_RUNTIME_MODES -DHGTXR_E2E_TEST_RUNTIME_MODES=0 -DHGTXR_E2E_FORCE_RUNTIME_MODE=0 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar4_w2bank8_tok2_attntok2_attnbram_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_track_only)
    PROJECT="hgtxr_e2e_axis_par32_rt_attntok2_attnbram_hpar4_w2b8_300_mem16_track_only_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_URAM_ATTN -DHGTXR_E2E_URAM_ATTN=0 -UHGTXR_E2E_LUTRAM_ATTN -DHGTXR_E2E_LUTRAM_ATTN=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=4 -DHGTXR_E2E_PATCH_PAR=16 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -DHGTXR_E2E_MLP_TOKEN_PAR=2 -DHGTXR_E2E_DENSE_TOKEN_PAR=2 -DHGTXR_E2E_MLP_FUSED_W2=1 -DHGTXR_E2E_MLP_FUSED_W2_BANKED_ACC=1 -DHGTXR_E2E_MLP_FUSED_W2_HP_PAR=4 -DHGTXR_E2E_W2_WEIGHT_CACHE_BANKS=8 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -UHGTXR_E2E_TEST_RUNTIME_MODES -DHGTXR_E2E_TEST_RUNTIME_MODES=0 -DHGTXR_E2E_FORCE_RUNTIME_MODE=1 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_nohidden_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_nohidden_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=4 -DHGTXR_E2E_PATCH_PAR=16 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -DHGTXR_E2E_MLP_FUSED_W2=1 -DHGTXR_E2E_MLP_FUSED_W2_BANKED_ACC=1 -DHGTXR_E2E_MLP_FUSED_W2_SKIP_HIDDEN_STORE=1 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_nohidden_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_nohidden_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=4 -DHGTXR_E2E_PATCH_PAR=16 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -DHGTXR_E2E_MLP_FUSED_W2=1 -DHGTXR_E2E_MLP_FUSED_W2_BANKED_ACC=1 -DHGTXR_E2E_MLP_FUSED_W2_SKIP_HIDDEN_STORE=1 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -UHGTXR_E2E_TEST_RUNTIME_MODES -DHGTXR_E2E_TEST_RUNTIME_MODES=0 -DHGTXR_E2E_FORCE_RUNTIME_MODE=0 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_nohidden_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_track_only)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_nohidden_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_track_only_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=4 -DHGTXR_E2E_PATCH_PAR=16 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -DHGTXR_E2E_MLP_FUSED_W2=1 -DHGTXR_E2E_MLP_FUSED_W2_BANKED_ACC=1 -DHGTXR_E2E_MLP_FUSED_W2_SKIP_HIDDEN_STORE=1 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -UHGTXR_E2E_TEST_RUNTIME_MODES -DHGTXR_E2E_TEST_RUNTIME_MODES=0 -DHGTXR_E2E_FORCE_RUNTIME_MODE=1 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_dense64_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_dense64_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_DENSE_PAR -DHGTXR_E2E_DENSE_PAR=64 -UHGTXR_E2E_ATTN_PAR -DHGTXR_E2E_ATTN_PAR=64 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=4 -DHGTXR_E2E_PATCH_PAR=16 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_dense64_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_dense64_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_DENSE_PAR -DHGTXR_E2E_DENSE_PAR=64 -UHGTXR_E2E_ATTN_PAR -DHGTXR_E2E_ATTN_PAR=64 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=4 -DHGTXR_E2E_PATCH_PAR=16 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -UHGTXR_E2E_TEST_RUNTIME_MODES -DHGTXR_E2E_TEST_RUNTIME_MODES=0 -DHGTXR_E2E_FORCE_RUNTIME_MODE=0 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_dense64_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_track_only)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_dense64_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_track_only_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_OBSERVE_DATAPATH -DHGTXR_E2E_OBSERVE_DATAPATH=0 -UHGTXR_E2E_URAM_NORM -DHGTXR_E2E_URAM_NORM=0 -UHGTXR_E2E_LUTRAM_NORM -DHGTXR_E2E_LUTRAM_NORM=0 -UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -UHGTXR_E2E_QKV_WEIGHT_CACHE -DHGTXR_E2E_QKV_WEIGHT_CACHE=0 -UHGTXR_E2E_LN_PARAM_CACHE -DHGTXR_E2E_LN_PARAM_CACHE=0 -UHGTXR_E2E_DENSE_PAR -DHGTXR_E2E_DENSE_PAR=64 -UHGTXR_E2E_ATTN_PAR -DHGTXR_E2E_ATTN_PAR=64 -UHGTXR_E2E_HEAD_PAR -DHGTXR_E2E_HEAD_PAR=4 -DHGTXR_E2E_PATCH_PAR=16 -DHGTXR_E2E_PATCH_TOKEN_LOOP=1 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -UHGTXR_E2E_TEST_RUNTIME_MODES -DHGTXR_E2E_TEST_RUNTIME_MODES=0 -DHGTXR_E2E_FORCE_RUNTIME_MODE=1 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_dsppipe4_300_mem16)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_dsppipe4_300_mem16_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL -DHGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=0 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_SHARE_RUNTIME_UNITS=1 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail16_softmaxq_lutbuf_dsppipe4_300_mem16)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail16_softmaxq_lutbuf_dsppipe4_300_mem16_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_USE_HGPIPE_INT_SOFTMAXQ=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_CORE_FABRIC_TAIL_LANES=16 -DHGTXR_E2E_CORE_LANE_CT_SWITCH=1 -DHGTXR_E2E_LUTRAM_FRAME_TOKENS=1 -DHGTXR_E2E_LUTRAM_GB_TOKENS=1 -DHGTXR_E2E_LUTRAM_NORM=1 -DHGTXR_E2E_LUTRAM_Q=1 -DHGTXR_E2E_LUTRAM_K=1 -DHGTXR_E2E_LUTRAM_V=1 -DHGTXR_E2E_LUTRAM_ATTN=1 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail16_intnl_lutbuf_acc24_dsppipe4_300_mem16)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail16_intnl_lutbuf_acc24_dsppipe4_300_mem16_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_ACC_W -DHGTXR_ACC_W=24 -UHGTXR_ACC_I -DHGTXR_ACC_I=10 -DHGTXR_E2E_USE_HGPIPE_INT_GELUQ=1 -DHGTXR_E2E_USE_HGPIPE_INT_SOFTMAXQ=1 -DHGTXR_E2E_USE_HGPIPE_INT_LAYERNORMQ=1 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_CORE_FABRIC_TAIL_LANES=16 -DHGTXR_E2E_CORE_LANE_CT_SWITCH=1 -DHGTXR_E2E_LUTRAM_FRAME_TOKENS=1 -DHGTXR_E2E_LUTRAM_GB_TOKENS=1 -DHGTXR_E2E_LUTRAM_NORM=1 -DHGTXR_E2E_LUTRAM_Q=1 -DHGTXR_E2E_LUTRAM_K=1 -DHGTXR_E2E_LUTRAM_V=1 -DHGTXR_E2E_LUTRAM_ATTN=1 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_coreallfabric_nowide_lutbuf_300_mem16)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_coreallfabric_nowide_lutbuf_300_mem16_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_FORCE_DSP_MUL -DHGTXR_E2E_FORCE_DSP_MUL=0 -UHGTXR_E2E_FORCE_WIDE_DSP_MUL -DHGTXR_E2E_FORCE_WIDE_DSP_MUL=0 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_CORE_ALL_FABRIC_MUL=1 -DHGTXR_E2E_LUTRAM_FRAME_TOKENS=1 -DHGTXR_E2E_LUTRAM_GB_TOKENS=1 -DHGTXR_E2E_LUTRAM_NORM=1 -DHGTXR_E2E_LUTRAM_Q=1 -DHGTXR_E2E_LUTRAM_K=1 -DHGTXR_E2E_LUTRAM_V=1 -DHGTXR_E2E_LUTRAM_ATTN=1 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_coreallfabric_intnl_lutbuf_acc24_300_mem16)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_coreallfabric_intnl_lutbuf_acc24_300_mem16_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_ACC_W -DHGTXR_ACC_W=24 -UHGTXR_ACC_I -DHGTXR_ACC_I=10 -UHGTXR_E2E_FORCE_DSP_MUL -DHGTXR_E2E_FORCE_DSP_MUL=0 -UHGTXR_E2E_FORCE_WIDE_DSP_MUL -DHGTXR_E2E_FORCE_WIDE_DSP_MUL=0 -DHGTXR_E2E_USE_HGPIPE_INT_GELUQ=1 -DHGTXR_E2E_USE_HGPIPE_INT_SOFTMAXQ=1 -DHGTXR_E2E_USE_HGPIPE_INT_LAYERNORMQ=1 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_CORE_ALL_FABRIC_MUL=1 -DHGTXR_E2E_LUTRAM_FRAME_TOKENS=1 -DHGTXR_E2E_LUTRAM_GB_TOKENS=1 -DHGTXR_E2E_LUTRAM_NORM=1 -DHGTXR_E2E_LUTRAM_Q=1 -DHGTXR_E2E_LUTRAM_K=1 -DHGTXR_E2E_LUTRAM_V=1 -DHGTXR_E2E_LUTRAM_ATTN=1 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par16_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_toklutbuf_intnl_acc24_dsppipe4_300_mem8)
    PROJECT="hgtxr_e2e_axis_par16_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_toklutbuf_intnl_acc24_dsppipe4_300_mem8_no_board"
    E2E_PAR=16
    MEM_BANK_PAR=8
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_ACC_W -DHGTXR_ACC_W=24 -UHGTXR_ACC_I -DHGTXR_ACC_I=10 -DHGTXR_E2E_USE_HGPIPE_INT_GELUQ=1 -DHGTXR_E2E_USE_HGPIPE_INT_SOFTMAXQ=1 -DHGTXR_E2E_USE_HGPIPE_INT_LAYERNORMQ=1 -DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_CORE_FABRIC_TAIL_LANES=2 -DHGTXR_E2E_CORE_LANE_CT_SWITCH=1 -DHGTXR_E2E_LUTRAM_FRAME_TOKENS=1 -DHGTXR_E2E_LUTRAM_GB_TOKENS=1 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_dsppipe4_keep_300_mem16)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_dsppipe4_keep_300_mem16_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute_keep
    EXTRA_CFLAGS="-DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -DHGTXR_E2E_NONLINEAR_ROM_LUTRAM=1 -DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1 -DHGTXR_E2E_DSP_MUL_LATENCY=4 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_CORE_FABRIC_TAIL_LANES=2 -DHGTXR_E2E_CORE_LANE_CT_SWITCH=1 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1 -DHGTXR_HLS_FIXED_CSIM=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_300_mem16_search_only)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_300_mem16_search_only_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-DHGTXR_E2E_DISPATCH_PREFETCH_BANKS=4 -DHGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1 -UHGTXR_E2E_TEST_RUNTIME_MODES -DHGTXR_E2E_TEST_RUNTIME_MODES=0 -DHGTXR_E2E_FORCE_RUNTIME_MODE=0 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1 -DHGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_300_mem16_search_only)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_300_mem16_search_only_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_TEST_RUNTIME_MODES -DHGTXR_E2E_TEST_RUNTIME_MODES=0 -DHGTXR_E2E_FORCE_RUNTIME_MODE=0 -DHGTXR_E2E_CSIM_PREFETCH_TRACE=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_300_mem16_track_only)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_300_mem16_track_only_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_TEST_RUNTIME_MODES -DHGTXR_E2E_TEST_RUNTIME_MODES=0 -DHGTXR_E2E_FORCE_RUNTIME_MODE=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_pipepref_300_mem16)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_pipepref_300_mem16_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-DHGTXR_E2E_STAGE_PREFETCH_LOAD=1 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_pipepref_tail4_nowide_300_mem16)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_pipepref_tail4_nowide_300_mem16_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=compute
    EXTRA_CFLAGS="-UHGTXR_E2E_FORCE_WIDE_DSP_MUL -DHGTXR_E2E_FORCE_WIDE_DSP_MUL=0 -DHGTXR_E2E_STAGE_PREFETCH_LOAD=1 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_CORE_FABRIC_TAIL_LANES=4 -DHGTXR_E2E_CORE_LANE_CT_SWITCH=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_pipepref_300_mem16)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_pipepref_300_mem16_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=datapath
    EXTRA_CFLAGS="-DHGTXR_E2E_STAGE_PREFETCH_LOAD=1 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_pipepref_tail2_300_mem16)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_pipepref_tail2_300_mem16_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=datapath
    EXTRA_CFLAGS="-DHGTXR_E2E_STAGE_PREFETCH_LOAD=1 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_CORE_FABRIC_TAIL_LANES=2"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_pipepref_tail4_300_mem16)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_pipepref_tail4_300_mem16_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=datapath
    EXTRA_CFLAGS="-DHGTXR_E2E_STAGE_PREFETCH_LOAD=1 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_CORE_FABRIC_TAIL_LANES=4"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_pipepref_cttail4_300_mem16)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_pipepref_cttail4_300_mem16_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=datapath
    EXTRA_CFLAGS="-DHGTXR_E2E_STAGE_PREFETCH_LOAD=1 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_CORE_FABRIC_TAIL_LANES=4 -DHGTXR_E2E_CORE_LANE_CT_SWITCH=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_pipepref_tail8_300_mem16)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_pipepref_tail8_300_mem16_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=datapath
    EXTRA_CFLAGS="-DHGTXR_E2E_STAGE_PREFETCH_LOAD=1 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_CORE_FABRIC_TAIL_LANES=8"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_pipepref_coreallfabric_300_mem16)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_pipepref_coreallfabric_300_mem16_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=datapath
    EXTRA_CFLAGS="-DHGTXR_E2E_STAGE_PREFETCH_LOAD=1 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_CORE_ALL_FABRIC_MUL=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_pipepref_coreallfabric_nowide_300_mem16)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_pipepref_coreallfabric_nowide_300_mem16_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=datapath
    EXTRA_CFLAGS="-UHGTXR_E2E_FORCE_WIDE_DSP_MUL -DHGTXR_E2E_FORCE_WIDE_DSP_MUL=0 -DHGTXR_E2E_STAGE_PREFETCH_LOAD=1 -DHGTXR_E2E_FABRIC_MUL_LATENCY=3 -DHGTXR_E2E_CORE_ALL_FABRIC_MUL=1"
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_topdt_300_mem16)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_topdt_300_mem16_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=top
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_nodt_300_mem16)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_nodt_300_mem16_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=0
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normstage_300_mem16)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normstage_300_mem16_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_normstream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normstage_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=datapath
    ;;
  par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_noobs_300_mem16)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_noobs_300_mem16_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_stream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_headraw_addrsw_noobs_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=datapath
    ;;
  par32_runtime_rom_only_dispatch_dsp3_intnl_headraw_addrsw_300_mem16)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_intnl_headraw_addrsw_300_mem16_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_stream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_intnl_headraw_addrsw_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=datapath
    ;;
  par32_runtime_rom_only_dispatch_dsp3_300_mem16_search_only)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_300_mem16_search_only_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_stream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_active64_b8_ff768_search_only"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=datapath
    ;;
  par32_runtime_rom_only_dispatch_dsp3_300_mem16_track_only)
    PROJECT="hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_300_mem16_track_only_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_stream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_active64_b8_ff768_track_only"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=datapath
    ;;
  par16_runtime_rom_dispatch_300_mem16)
    PROJECT="hgtxr_e2e_axis_par16_runtime_rom_dispatch_300_mem16_no_board"
    E2E_PAR=16
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_stream"
    E2E_SCALE="runtime_rom_dispatch_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    ;;
  par8_runtime_rom_dispatch_300_mem8)
    PROJECT="hgtxr_e2e_axis_par8_runtime_rom_dispatch_300_mem8_no_board"
    E2E_PAR=8
    MEM_BANK_PAR=8
    RESOURCE_POLICY="dsp_mixed_stream"
    E2E_SCALE="runtime_rom_dispatch_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=datapath
    ;;
  par8_runtime_rom_only_dispatch_300_mem8)
    PROJECT="hgtxr_e2e_axis_par8_runtime_rom_only_dispatch_300_mem8_no_board"
    E2E_PAR=8
    MEM_BANK_PAR=8
    RESOURCE_POLICY="dsp_mixed_stream"
    E2E_SCALE="runtime_rom_only_dispatch_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=datapath
    ;;
  par8_runtime_rom_only_dispatch_dsp3_300_mem8)
    PROJECT="hgtxr_e2e_axis_par8_runtime_rom_only_dispatch_dsp3_300_mem8_no_board"
    E2E_PAR=8
    MEM_BANK_PAR=8
    RESOURCE_POLICY="dsp_mixed_stream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_active64_b8_ff768"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=datapath
    ;;
  par8_runtime_rom_only_dispatch_dsp3_300_mem8_search_only)
    PROJECT="hgtxr_e2e_axis_par8_runtime_rom_only_dispatch_dsp3_300_mem8_search_only_no_board"
    E2E_PAR=8
    MEM_BANK_PAR=8
    RESOURCE_POLICY="dsp_mixed_stream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_active64_b8_ff768_search_only"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=datapath
    ;;
  par8_runtime_rom_only_dispatch_dsp3_300_mem8_track_only)
    PROJECT="hgtxr_e2e_axis_par8_runtime_rom_only_dispatch_dsp3_300_mem8_track_only_no_board"
    E2E_PAR=8
    MEM_BANK_PAR=8
    RESOURCE_POLICY="dsp_mixed_stream"
    E2E_SCALE="runtime_rom_only_dispatch_dsp3_active64_b8_ff768_track_only"
    HLS_CLOCK_PERIOD=3.333
    E2E_OOC_DONT_TOUCH=datapath
    ;;
  par32_dsp_uram_mem16)
    PROJECT="hgtxr_e2e_axis_par32_dsp_uram_mem16_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_uram"
    E2E_SCALE="full"
    ;;
  par32_dsp_uram_mem32)
    PROJECT="hgtxr_e2e_axis_par32_dsp_uram_mem32_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=32
    RESOURCE_POLICY="dsp_uram"
    E2E_SCALE="full"
    ;;
  par32_dsp_mixed_stream_mem16)
    PROJECT="hgtxr_e2e_axis_par32_dsp_mixed_stream_mem16_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed_stream"
    E2E_SCALE="full"
    ;;
  par32_dsp_mixed_mem16)
    PROJECT="hgtxr_e2e_axis_par32_dsp_mixed_mem16_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=16
    RESOURCE_POLICY="dsp_mixed"
    E2E_SCALE="full"
    ;;
  par32_dsp_mixed_mem32)
    PROJECT="hgtxr_e2e_axis_par32_dsp_mixed_mem32_no_board"
    E2E_PAR=32
    MEM_BANK_PAR=32
    RESOURCE_POLICY="dsp_mixed"
    E2E_SCALE="full"
    ;;
  *)
    echo "unknown profile: $PROFILE" >&2
    echo "additional profile: par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail4_dsppipe4_300_mem16" >&2
    echo "additional profile: par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail8_lutbuf_dsppipe4_300_mem16" >&2
    echo "additional profile: par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail16_lutbuf_dsppipe4_300_mem16" >&2
    echo "additional profile: par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail16_lutbuf_shareunit_dsppipe4_300_mem16" >&2
    echo "additional profile: par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_lutbuf_shareunit_dsppipe4_300_mem16" >&2
    echo "additional profile: par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_lutbuf_shareunit_headpar8_dsppipe4_300_mem16" >&2
    echo "additional profile: par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_dsppipe4_300_mem16" >&2
    echo "additional profile: par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_dsppipe4_300_mem16" >&2
    echo "additional profile: par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_dsppipe4_300_mem16_search_only" >&2
    echo "additional profile: par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_dsppipe4_300_mem16_track_only" >&2
    echo "additional profile: par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_dsppipe4_300_mem16" >&2
    echo "additional profile: par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail16_softmaxq_lutbuf_dsppipe4_300_mem16" >&2
    echo "additional profile: par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail16_intnl_lutbuf_acc24_dsppipe4_300_mem16" >&2
    echo "additional profile: par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_coreallfabric_nowide_lutbuf_300_mem16" >&2
    echo "additional profile: par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_coreallfabric_intnl_lutbuf_acc24_300_mem16" >&2
    echo "additional profile: par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_toklutbuf_dsppipe4_300_mem16" >&2
    echo "additional profile: par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_toklutbuf_auxfabric_normlut_dsppipe4_300_mem16" >&2
    echo "additional profile: par16_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_toklutbuf_intnl_acc24_dsppipe4_300_mem8" >&2
    echo "additional profile: par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_dsppipe4_keep_300_mem16" >&2
    echo "profiles: par16_c3b_recheck par32_runtime_mode_mem16 par32_runtime_full_axi_mem16 par32_runtime_full_axi_dsp_mem16 par32_runtime_rom_dispatch_300_mem16 par32_runtime_rom_only_dispatch_dsp3_300_mem16 par32_runtime_rom_only_dispatch_dsp3_headlut_300_mem16 par32_runtime_rom_only_dispatch_dsp3_addrsw_300_mem16 par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_300_mem16 par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_300_mem16 par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_300_mem16 par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_300_mem16 par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_300_mem16 par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_300_mem16 par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_dsppipe4_300_mem16 par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_toklutbuf_dsppipe4_300_mem16 par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_300_mem16_search_only par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_300_mem16_search_only par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_300_mem16_track_only par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_pipepref_300_mem16 par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_pipepref_tail2_300_mem16 par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_pipepref_tail4_300_mem16 par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_pipepref_cttail4_300_mem16 par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_pipepref_tail8_300_mem16 par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_pipepref_coreallfabric_300_mem16 par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_pipepref_coreallfabric_nowide_300_mem16 par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_topdt_300_mem16 par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_nodt_300_mem16 par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normstage_300_mem16 par32_runtime_rom_only_dispatch_dsp3_intnl_headraw_addrsw_300_mem16 par32_runtime_rom_only_dispatch_dsp3_300_mem16_search_only par32_runtime_rom_only_dispatch_dsp3_300_mem16_track_only par16_runtime_rom_dispatch_300_mem16 par8_runtime_rom_dispatch_300_mem8 par8_runtime_rom_only_dispatch_300_mem8 par8_runtime_rom_only_dispatch_dsp3_300_mem8 par8_runtime_rom_only_dispatch_dsp3_300_mem8_search_only par8_runtime_rom_only_dispatch_dsp3_300_mem8_track_only par32_dsp_uram_mem16 par32_dsp_uram_mem32 par32_dsp_mixed_stream_mem16 par32_dsp_mixed_mem16 par32_dsp_mixed_mem32" >&2
    exit 2
    ;;
esac

if [ -n "${HGTXR_E2E_PROJECT_NAME:-}" ]; then
  PROJECT="$HGTXR_E2E_PROJECT_NAME"
fi
if [ -n "${HGTXR_E2E_PAR:-}" ]; then
  E2E_PAR="$HGTXR_E2E_PAR"
fi
if [ -n "${HGTXR_E2E_MEM_BANK_PAR:-}" ]; then
  MEM_BANK_PAR="$HGTXR_E2E_MEM_BANK_PAR"
fi
if [ -n "${HGTXR_E2E_RESOURCE_POLICY:-}" ]; then
  RESOURCE_POLICY="$HGTXR_E2E_RESOURCE_POLICY"
fi
if [ -n "${HGTXR_E2E_SCALE:-}" ]; then
  E2E_SCALE="$HGTXR_E2E_SCALE"
fi
if [ -n "${HGTXR_HLS_CLOCK_PERIOD:-}" ]; then
  HLS_CLOCK_PERIOD="$HGTXR_HLS_CLOCK_PERIOD"
fi

HLS_BIN=${VITIS_HLS_BIN:-}
if [ -z "$HLS_BIN" ]; then
  if command -v vitis_hls >/dev/null 2>&1; then
    HLS_BIN=$(command -v vitis_hls)
  elif [ -x /tools/Xilinx/Vitis_HLS/2023.2/bin/vitis_hls ]; then
    HLS_BIN=/tools/Xilinx/Vitis_HLS/2023.2/bin/vitis_hls
  else
    echo "vitis_hls not found. Set VITIS_HLS_BIN or source Xilinx environment." >&2
    exit 127
  fi
fi

XILINX_ROOT=${HGTXR_XILINX_ROOT:-/tools/Xilinx}
VITIS_HLS_LIB_DIR="$XILINX_ROOT/Vitis_HLS/2023.2/lib/lnx64.o/Rhel/9"
if [ -d "$VITIS_HLS_LIB_DIR" ]; then
  export LD_LIBRARY_PATH="$VITIS_HLS_LIB_DIR:${LD_LIBRARY_PATH:-}"
fi

REPORT_DIR="generated/$PROJECT/solution_e2e_q4w8a/syn/report"

echo "mode=$MODE"
echo "profile=$PROFILE"
echo "project=$PROJECT"
echo "scale=$E2E_SCALE"
echo "parallelism=$E2E_PAR"
echo "mem_bank_par=$MEM_BANK_PAR"
echo "resource_policy=$RESOURCE_POLICY"
echo "extra_cflags=$EXTRA_CFLAGS"
echo "hls_clock_period=${HLS_CLOCK_PERIOD:-5.0}"
echo "ooc_dont_touch=${E2E_OOC_DONT_TOUCH:-${HGTXR_E2E_OOC_DONT_TOUCH:-0}}"
echo "hls_bin=$HLS_BIN"
echo "tcl=$HLS_TCL"
echo "vitis_hls_lib_dir=$VITIS_HLS_LIB_DIR"
if [ "$MODE" = "csynth" ]; then
  echo "expected_report=$REPORT_DIR/hgtxr_e2e_axis_top_csynth.rpt"
elif [ "$MODE" = "package" ]; then
  echo "expected_ip=generated/$PROJECT/solution_e2e_q4w8a/impl/ip/component.xml"
fi

export HGTXR_E2E_PROJECT_NAME="$PROJECT"
export HGTXR_E2E_SCALE="$E2E_SCALE"
export HGTXR_E2E_PAR="$E2E_PAR"
export HGTXR_E2E_MEM_BANK_PAR="$MEM_BANK_PAR"
export HGTXR_E2E_RESOURCE_POLICY="$RESOURCE_POLICY"
export HGTXR_E2E_EXTRA_CFLAGS="${HGTXR_E2E_EXTRA_CFLAGS:-$EXTRA_CFLAGS}"
export HGTXR_HLS_CLOCK_PERIOD="${HLS_CLOCK_PERIOD:-5.0}"
export HGTXR_E2E_OOC_DONT_TOUCH="${HGTXR_E2E_OOC_DONT_TOUCH:-${E2E_OOC_DONT_TOUCH:-0}}"

exec "$HLS_BIN" -f "$HLS_TCL"
