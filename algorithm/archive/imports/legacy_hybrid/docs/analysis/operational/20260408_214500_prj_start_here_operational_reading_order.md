# 프로젝트 시작 문서

이 문서는 `HBTXR_v3_0`에서 운영자나 구현자가 가장 먼저 읽어야 하는 문서만 추려 둔 빠른 시작 안내입니다.

## 1. 먼저 보는 문서

일상적인 실행, 디버깅, 구조 파악은 아래 순서로 읽는 것을 권장합니다.

1. [README.md](../../README.md)
2. [20260330_170419_06_v3_repository_tree_and_module_dependency_map.md](20260330_170419_06_v3_repository_tree_and_module_dependency_map.md)
3. [20260408_233000_v3_project_structure_and_pipeline_analysis.md](20260408_233000_v3_project_structure_and_pipeline_analysis.md)
4. [20260408_235400_v3_training_evaluation_metrics_and_deployment_guide.md](20260408_235400_v3_training_evaluation_metrics_and_deployment_guide.md)
5. [20260408_235500_external_package_operating_guide.md](20260408_235500_external_package_operating_guide.md)
6. [20260409_001000_mode_preprocess_step_by_step_execution_guide.md](20260409_001000_mode_preprocess_step_by_step_execution_guide.md)
7. [20260410_110000_timelens_xl_finetuning_workflow.md](20260410_110000_timelens_xl_finetuning_workflow.md)
8. [20260410_140000_v2e_event_generation_experiment_workflow.md](20260410_140000_v2e_event_generation_experiment_workflow.md)
9. [20260410_170000_timelens_xl_and_v2e_experiment_status_report.md](../others/update/20260410_170000_timelens_xl_and_v2e_experiment_status_report.md)
10. [20260330_170419_10_v3_yaml_config_reference.md](20260330_170419_10_v3_yaml_config_reference.md)
11. [20260330_170419_09_v3_mode_data_pipeline_contracts.md](20260330_170419_09_v3_mode_data_pipeline_contracts.md)
12. [20260408_190000_tracker_dataset_output_contracts.md](20260408_190000_tracker_dataset_output_contracts.md)
13. [20260408_233200_train_hbtxr_actual_execution_flow.md](20260408_233200_train_hbtxr_actual_execution_flow.md)
14. [20260408_233400_dataset_model_loss_detailed_call_stack.md](20260408_233400_dataset_model_loss_detailed_call_stack.md)

## 2. 질문별 추천 문서

- 저장소 구조를 알고 싶다
  - [20260330_170419_06_v3_repository_tree_and_module_dependency_map.md](20260330_170419_06_v3_repository_tree_and_module_dependency_map.md)
- 전체 구조와 공식 파이프라인을 한 번에 이해하고 싶다
  - [20260408_233000_v3_project_structure_and_pipeline_analysis.md](20260408_233000_v3_project_structure_and_pipeline_analysis.md)
- 학습/평가 메트릭과 배포 스크립트까지 한 번에 보고 싶다
  - [20260408_235400_v3_training_evaluation_metrics_and_deployment_guide.md](20260408_235400_v3_training_evaluation_metrics_and_deployment_guide.md)
- 외부 패키지, backend selector, package hub 운영 정책을 알고 싶다
  - [20260408_235500_external_package_operating_guide.md](20260408_235500_external_package_operating_guide.md)
- config와 preset 차이를 알고 싶다
  - [20260330_170419_10_v3_yaml_config_reference.md](20260330_170419_10_v3_yaml_config_reference.md)
- mode1/mode2 데이터 계약 차이를 알고 싶다
  - [20260330_170419_09_v3_mode_data_pipeline_contracts.md](20260330_170419_09_v3_mode_data_pipeline_contracts.md)
- Grounded-SAM부터 manifest까지 파이프라인을 알고 싶다
  - [20260330_170419_03_groundedsam_dataset_pipeline.md](20260330_170419_03_groundedsam_dataset_pipeline.md)
- mode별 pre-process를 실제 명령 순서대로 보고 싶다
  - [20260409_001000_mode_preprocess_step_by_step_execution_guide.md](20260409_001000_mode_preprocess_step_by_step_execution_guide.md)
- TimeLens-XL fine-tuning export/launcher 흐름을 알고 싶다
  - [20260410_110000_timelens_xl_finetuning_workflow.md](20260410_110000_timelens_xl_finetuning_workflow.md)
- `v2e`를 baseline과 나란히 비교 실험하고 싶다
  - [20260410_140000_v2e_event_generation_experiment_workflow.md](20260410_140000_v2e_event_generation_experiment_workflow.md)
- 최근 `TimeLens-XL` / `v2e` 실험 현황과 남은 작업을 한 번에 보고 싶다
  - [20260410_170000_timelens_xl_and_v2e_experiment_status_report.md](../others/update/20260410_170000_timelens_xl_and_v2e_experiment_status_report.md)
- `train_hbtxr.py`가 실제로 어떤 순서로 실행되는지 알고 싶다
  - [20260408_233200_train_hbtxr_actual_execution_flow.md](20260408_233200_train_hbtxr_actual_execution_flow.md)
- `dataset -> model -> loss` 호출 스택을 함수 단위로 보고 싶다
  - [20260408_233400_dataset_model_loss_detailed_call_stack.md](20260408_233400_dataset_model_loss_detailed_call_stack.md)
- tracker/dataset 출력 key와 snapshot 기준을 알고 싶다
  - [20260408_190000_tracker_dataset_output_contracts.md](20260408_190000_tracker_dataset_output_contracts.md)
- 모델 구조와 head/loss 연결을 더 깊게 보고 싶다
  - [20260330_170419_01_v3_code_architecture_analysis.md](20260330_170419_01_v3_code_architecture_analysis.md)
  - [20260330_170419_08_v3_heads_outputs_and_stage_losses.md](20260330_170419_08_v3_heads_outputs_and_stage_losses.md)

## 3. 문서 경계

- `docs/prj/`
  - 현재 프로젝트를 운영하고 수정하는 데 필요한 문서를 우선 배치합니다.
- `docs/exps/`
  - 실험 결과, 진행 로그, 비교 분석을 둡니다.
- `docs/others/analysis/`
  - 긴 조사 문서나 보조 분석 문서를 보관합니다.
- `docs/others/update/`
  - 구조 개편, 병합, 마이그레이션 같은 업데이트 기록을 둡니다.

## 4. 현재 공식 실행 표면

- main runtime surface
  - `scripts/`
  - `configs/base.yaml`
  - `configs/mode1_stage1.yaml`
  - `configs/mode1_stage2.yaml`
  - `configs/mode2_stage1.yaml`
  - `configs/mode2_stage2.yaml`
- experimental surface
  - `exps/scripts/`
  - `exps/configs/`

실험 preset이나 one-off helper를 찾고 있다면 먼저 `docs/exps/`와 `exps/README.md`를 확인하는 편이 빠릅니다.
