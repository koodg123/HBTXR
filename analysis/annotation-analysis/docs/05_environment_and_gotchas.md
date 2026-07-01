# 05 · 실행 환경 & 주의점 (Gotchas)

## 환경 제약
- **[2026-07-01 정정] 현재 WSL 네이티브 CLI 세션은 bash 정상 동작** — GSAM2(08)를 세션 내에서 직접 실행 성공(GPU 포함). 아래 "bash 불가"는 **이전 Cowork/Desktop UNC 세션 한정**. GSAM2 환경 레시피·실행법은 [07](07_gsam2_audit_results.md) 참고.
- **Cowork 세션 bash 불가**: 연결 폴더가 `\\wsl.localhost\...` **UNC**라 리눅스 샌드박스가 마운트 실패("UNC paths are not supported") → **모든 bash/tmux/uv/모델 실행이 세션 내에서 불가**. 파일 도구(Read/Write/Glob)만 동작.
  - Claude Desktop 셸은 Windows에서 실행되고, 샌드박스는 macOS/Linux/WSL2만 지원(네이티브 Windows 미지원).
  - **네이티브 경로+bash를 쓰려면**: WSL 터미널에서 `cd /home/user/project/.../HBTXR && claude`(CLI 실행) → cwd가 네이티브 `/home/...`, 샌드박스(bubblewrap/socat) 동작. 단 Cowork GUI 대신 CLI.
  - 또는 프로젝트를 윈도우 드라이브(E:/C:)로 복사해 연결.
- **무거운 실행은 사용자가 WSL tmux에서**: GSAM2(08), HBTXR(09), uv 등.
- **Glob 주의**: 거대 폴더(dataset) top-level `*`는 타임아웃. `path`를 정확한 하위폴더로 주고 슬래시 없는 패턴 사용.

## 발견/수정된 버그
### B1. events.txt 필드 오해 (수정됨)
- `events.txt`는 **4필드 `ts x y pol`**(맨 앞이 timestamp), 5필드 아님.
- 원인: Read 도구의 **줄번호 접두어**를 데이터 index로 오독.
- 영향: `slice_events`가 `len<5`로 전 줄 skip → **n_events=0**.
- 수정: `evlib.parse_events_line`, `07 slice_events`를 4필드로. `07b_reslice_events.py`로 이벤트만 재생성.

### B2. 모션 오분류 (수정됨)
- 초기 vfix0.6/max-velocity → fixation18/saccade74(역전). 블링크 centroid 점프가 saccade로 오분류.
- 수정: 블링크 제외 + fixation=median 속도 + vfix1.2/vsac6 → fixation306/saccade161(정상).

## 주의점 (분석 시)
### G1. y_unet 출처 불일치
- E: 사본의 dense U-Net = **EV-Eye 공식 `Data_davis_predict`**. 0.1812를 만든 건 **학습머신의 `DeanDataset_full_unet`**(다른 U-Net).
- → E: 기준 label-noise(orig vs unet)는 **근사**. 정확한 dense-label 대비 수치는 full_unet의 anchor 라벨을 붙여 `y_unet` 교체 필요.

### G2. 해상도 매핑 (09)
- 모델 center는 grid(=img_size/patch_size)에서 나옴. 346×260 환산 = `x*346/G, y*260/G`.
- config별 G: `full_unet`(256/4)=**64**, `img64_patch4`(64/4)=**16**, `img128_patch4`(128/4)=**32**. **ckpt↔config↔`--img-size/--patch-size` 일치 필수**(안 맞으면 좌표 어긋남).

### G3. subject-independent ckpt
- 09는 users 1–10을 **test 제외한** ckpt로. 포함된 ckpt면 E_i 낙관 편향.

### G4. Grounded-SAM2 검출률
- 근적외 동공은 흔한 클래스가 아니라 검출 실패(blink/속눈썹) 가능 → valid rate 확인, 낮으면 `--box-thr/--text-thr`↓ 또는 `--prompt "pupil. black pupil."`.

### G5. GSAM2 정밀도 proxy 한계
- 사람 재주석 불가라 precision은 **자동 proxy**(fixation F2F jitter + GSAM2 perturbation). 리벗에 "automated proxy, not human precision" 명시.

### G6. blink/occlusion 정책
- `close`/마스크 면적 붕괴 프레임은 `E_i`·`U_i`·precision·valid-rate에서 **동일 기준 제외**. 안 그러면 비율 왜곡.

### G7. bootstrap 단위
- CI는 **subject-level cluster bootstrap**(프레임 단위는 상관성으로 CI 과소).
