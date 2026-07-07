# 문서 구조 재편 마이그레이션 매니페스트

이 문서는 `docs/chat`, `docs/jetcas`를 제외한 전체 문서를 `update`, `experiments`, `analysis`, `plan`, `progress` 체계로 재분류하기 위한 기준표이자 현재 phase의 실행 기록입니다.

## 범위

- 유지:
  - `docs/chat/`
  - `docs/jetcas/`
- 재구성:
  - `docs/hbtxr/`
  - `docs/references_analysis/`
  - `docs/others/`
  - `docs/plan/`
  - `docs/prj/20260325_140031_00_docs_index.md`

## 공통 규칙

- 파일명은 가능한 한 유지합니다.
- 본문과 제목은 `docs/chat`, `docs/jetcas`를 제외하고 한국어로 전환합니다.
- 현재 phase에서는 새 canonical 폴더에 문서를 복제하고, 기존 경로는 호환 레이어로 유지합니다.
- `resources/` 자산은 참조 문서가 있는 canonical 폴더로 함께 복제합니다.

## 제외 폴더

| 경로 | 처리 |
| --- | --- |
| `docs/chat/` | 유지, 이동/번역 제외 |
| `docs/jetcas/` | 유지, 이동/번역 제외 |

## 상위 인덱스

| 현재 경로 | 목표 경로 | 분류 | 처리 |
| --- | --- | --- | --- |
| `docs/prj/20260325_140031_00_docs_index.md` | `docs/prj/20260325_140031_00_docs_index.md` | index | 새 구조 기준으로 갱신 |
| `docs/plan/00_index.md` | `docs/plan/00_index.md` | plan | 위치 유지, 한국어 전환 예정 |

## `docs/hbtxr/` 본문 문서 분류

| 현재 경로 | 목표 경로 | 분류 | 처리 방식 |
| --- | --- | --- | --- |
| `docs/prj/20260330_170419_01_v3_code_architecture_analysis.md` | `docs/prj/20260330_170419_01_v3_code_architecture_analysis.md` | analysis | canonical 복제 후 한국어 전환 |
| `docs/prj/20260330_170419_02_event_binning_and_accumulation_flow.md` | `docs/prj/20260330_170419_02_event_binning_and_accumulation_flow.md` | analysis | canonical 복제 후 한국어 전환 |
| `docs/prj/20260330_170419_03_groundedsam_dataset_pipeline.md` | `docs/prj/20260330_170419_03_groundedsam_dataset_pipeline.md` | analysis | canonical 복제 후 한국어 전환 |
| `docs/hbtxr/04_loss_fuction_all.md` | `docs/prj/20260330_170419_04_loss_fuction_all.md` | analysis | canonical 복제 후 한국어 전환 |
| `docs/hbtxr/05_loss_function_investigation.md` | `docs/prj/20260330_170419_05_loss_function_investigation.md` | analysis | canonical 복제 후 한국어 전환 |
| `docs/hbtxr/06_v3_repository_tree_and_module_dependency_map.md` | `docs/prj/20260330_170419_06_v3_repository_tree_and_module_dependency_map.md` | analysis | canonical 복제 후 한국어 전환 |
| `docs/prj/20260330_170419_07_ev_eye_dataset_analysis_results.md` | `docs/prj/20260330_170419_07_ev_eye_dataset_analysis_results.md` | analysis | canonical 복제 후 한국어 전환 |
| `docs/prj/20260330_170419_08_v3_heads_outputs_and_stage_losses.md` | `docs/prj/20260330_170419_08_v3_heads_outputs_and_stage_losses.md` | analysis | canonical 복제 후 한국어 전환 |
| `docs/prj/20260330_170419_09_v3_mode_data_pipeline_contracts.md` | `docs/prj/20260330_170419_09_v3_mode_data_pipeline_contracts.md` | analysis | canonical 복제 후 한국어 전환 |
| `docs/prj/20260330_170419_10_v3_yaml_config_reference.md` | `docs/prj/20260330_170419_10_v3_yaml_config_reference.md` | analysis | canonical 복제 후 한국어 전환 |
| `docs/prj/20260330_170419_11_v3_optimizer_pool_reference.md` | `docs/prj/20260330_170419_11_v3_optimizer_pool_reference.md` | analysis | canonical 복제 후 한국어 전환 |
| `docs/exps/20260330_170419_12_v3_mode2_optimizer_pool_cpu_pilot.md` | `docs/exps/20260330_170419_12_v3_mode2_optimizer_pool_cpu_pilot.md` | experiments | canonical 복제 후 한국어 전환 |
| `docs/exps/20260330_170419_13_v3_real_ev_eye_mode2_optimizer_pool_cpu_pilot.md` | `docs/exps/20260330_170419_13_v3_real_ev_eye_mode2_optimizer_pool_cpu_pilot.md` | experiments | canonical 복제 후 한국어 전환 |
| `docs/exps/20260330_170419_14_groundedsam_roi_crop_prompted_experiment_summary.md` | `docs/exps/20260330_170419_14_groundedsam_roi_crop_prompted_experiment_summary.md` | experiments | canonical 복제 후 한국어 전환 |
| `docs/exps/20260330_170419_15_all48_v2_baseline_failure_review.md` | `docs/exps/20260330_170419_15_all48_v2_baseline_failure_review.md` | experiments | canonical 복제 후 한국어 전환 |
| `docs/exps/20260330_170419_16_all48_v2_handoff.md` | `docs/exps/20260330_170419_16_all48_v2_handoff.md` | experiments | canonical 복제 후 한국어 전환 |
| `docs/exps/20260330_170419_17_all48_sampled_dataset_construction_experiment.md` | `docs/exps/20260330_170419_17_all48_sampled_dataset_construction_experiment.md` | experiments | canonical 복제 후 한국어 전환 |
| `docs/prj/20260330_170419_18_current_heuristic_eye_region_roi_algorithm.md` | `docs/prj/20260330_170419_18_current_heuristic_eye_region_roi_algorithm.md` | analysis | canonical 복제 후 한국어 전환 |
| `docs/exps/progress/20260330_170419_19_mode0_stage1_debug_history_and_progress.md` | `docs/exps/progress/20260330_170419_19_mode0_stage1_debug_history_and_progress.md` | progress | canonical 복제 후 한국어 전환 |
| `docs/others/update/20260330_170419_20_v3_selective_branch_integration_report.md` | `docs/others/update/20260330_170419_20_v3_selective_branch_integration_report.md` | update | canonical 복제 후 한국어 전환 |

## `docs/hbtxr/optimizers/` 분류

| 현재 경로 | 목표 경로 | 분류 | 처리 방식 |
| --- | --- | --- | --- |
| `docs/hbtxr/optimizers/*.md` | `docs/analysis/optimizers/*.md` | analysis | 디렉토리 단위 canonical 복제 후 한국어 전환 |

## `docs/hbtxr/resources/` 분류

| 현재 경로 | 목표 경로 | 분류 | 처리 방식 |
| --- | --- | --- | --- |
| `docs/hbtxr/resources/08_all48_v2_baseline_failure_review/` | `docs/experiments/resources/08_all48_v2_baseline_failure_review/` | experiments | canonical 복제 후 경로 정리 |

## `docs/references_analysis/` 분류

| 현재 경로 | 목표 경로 | 분류 | 처리 방식 |
| --- | --- | --- | --- |
| `docs/prj/references/20260330_170419_00_index.md` | `docs/prj/references/20260330_170419_00_index.md` | analysis | 디렉토리 기준 canonical 복제 |
| `docs/references_analysis/facet/00_index.md` | `docs/prj/references/facet/20260330_170419_00_index.md` | analysis | canonical 복제 후 한국어 전환 |
| `docs/prj/references/facet/20260330_170419_01_session_01_facet_paper_and_code_analysis.md` | `docs/prj/references/facet/20260330_170419_01_session_01_facet_paper_and_code_analysis.md` | analysis | canonical 복제 후 한국어 전환 |
| `docs/prj/references/facet/20260330_170419_02_session_02_tensor_dimension_trace.md` | `docs/prj/references/facet/20260330_170419_02_session_02_tensor_dimension_trace.md` | analysis | canonical 복제 후 한국어 전환 |
| `docs/prj/references/facet/20260330_170419_03_session_03_train_vs_infer_pipeline_table.md` | `docs/prj/references/facet/20260330_170419_03_session_03_train_vs_infer_pipeline_table.md` | analysis | canonical 복제 후 한국어 전환 |
| `docs/references_analysis/swift-eye/00_index.md` | `docs/prj/references/swift-eye/20260330_170419_00_index.md` | analysis | canonical 복제 후 한국어 전환 |
| `docs/prj/references/swift-eye/20260327_005435_01_swift_eye_timelens_pipeline_tables.md` | `docs/prj/references/swift-eye/20260327_005435_01_swift_eye_timelens_pipeline_tables.md` | analysis | canonical 복제 후 한국어 전환 |
| `docs/prj/references/swift-eye/20260330_170419_02_swift_eye_analysis_conversation_sessions.md` | `docs/prj/references/swift-eye/20260330_170419_02_swift_eye_analysis_conversation_sessions.md` | analysis | canonical 복제 후 한국어 전환 |
| `docs/prj/references/swift-eye/20260330_170419_03_swift_eye_paper_and_code_analysis.md` | `docs/prj/references/swift-eye/20260330_170419_03_swift_eye_paper_and_code_analysis.md` | analysis | canonical 복제 후 한국어 전환 |

## `docs/others/` 분류

| 현재 경로 | 목표 경로 | 분류 | 처리 방식 |
| --- | --- | --- | --- |
| `docs/others/00_index.md` | `docs/update/00_index.md` | update | update index로 흡수 |
| `docs/others/update/20260325_140031_02_legacy_archive_map.md` | `docs/others/update/20260325_140031_02_legacy_archive_map.md` | update | canonical 복제 후 한국어 전환 |

## `docs/plan/` 분류

| 현재 경로 | 목표 경로 | 분류 | 처리 방식 |
| --- | --- | --- | --- |
| `docs/plan/*.md` | `docs/plan/*.md` | plan | 위치 유지, 본문 한국어 전환 |

## 단계별 실행

1. 전수 분류 및 새 인덱스 골격 생성
2. 새 canonical 폴더 인덱스 확정
3. 문서와 자산 canonical 복제
4. 핵심 문서 링크 정리
5. 본문 한국어 전환
6. 최종 인덱스 / 링크 / 중복 검수

## 현재 상태

- 이번 phase에서 새 canonical 폴더에 실제 문서 복제를 수행했습니다.
- `docs/hbtxr`, `docs/references_analysis`, `docs/others`는 과거 링크 호환을 위한 레이어로 남겨둡니다.
- 핵심 인덱스와 대표 참조 문서의 경로를 새 canonical 구조 기준으로 일부 정리했습니다.
- 대량 한국어 전환은 아직 수행하지 않았습니다.
