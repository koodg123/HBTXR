> **작성** 2026-07-29 · **갱신** 2026-07-29
> **상태** active — M1~M5 진행에 따라 갱신
> **소유** hardware

# 코드 이관 대장 (P0)

상위 계획: [2026-07-29-hardware-reconstruction.md](2026-07-29-hardware-reconstruction.md) §6
데이터: [2026-07-29-code-migration-manifest.csv](2026-07-29-code-migration-manifest.csv) — **625행 전수**

## 왜 대장인가

원래 P0는 "죽은 것 삭제"였습니다. `hardware/`를 **제자리에서** 고친다는 전제였고,
전체를 `archive/`로 옮기고 새 트리를 채우는 방식으로 바뀌면서 무효가 됐습니다.
**`archive/`에서 지우지 않고, 안 옮기면 그만입니다.**

**그런데 안 옮긴 것은 눈에 보이지 않습니다.** "판단해서 뺐다"와 "빠뜨렸다"가
구분되지 않습니다. 그래서 P0는 삭제가 아니라 **전수 판정 기록**입니다.

> **이 대장이 즉시 값을 했습니다.** 만들자마자 **D1~D5 문서 이관의 구멍**을 찾아냈습니다 —
> 그 이관은 `.md`만 대상으로 해서, 캠페인 3개가 `data/` 없이 문서만 있었고 census
> 원자료 9개가 `archive/`에 남아 있었습니다. 11건 보완 완료.

## 판정 결과 — 625건

| 판정 | 수 | 뜻 |
|---|---:|---|
| `migrate` | **322** | 새 트리로 옮깁니다 |
| `done` | 234 | 이미 옮겼습니다 (문서 + 그 데이터) |
| `skip` | 64 | 판단해서 **안 옮깁니다** — 근거는 각 행에 |
| `undecided` | **5** | 조사 후 결정 |

`.pyc` 210개와 `.gitkeep` 29개는 결정 대상이 아니라 대장에서 제외했습니다 (864 − 239 = 625).

## `migrate` 322건의 목적지

| 목적지 | 수 | 단계 |
|---|---:|---|
| `hardware/tools/` | 87 | **M5** |
| `hardware/tools/tests/` | 60 | **M5** |
| `hardware/module/golden/` | 41 | **M2** |
| `hardware/config/` | 33 | **M1** |
| `hardware/build/vivado/` | 31 | **M3** |
| `hardware/module/include/` | 19 | **M2** |
| `hardware/module/src/` | 19 | **M2** |
| `hardware/module/tb/` | 12 | **M2** |
| `hardware/build/` | 11 | **M3** |
| `hardware/deploy/` | 9 | **M4** |

## `skip` 64건의 근거

| 무엇 | 근거 |
|---|---|
| `generated/**` | 빌드 산출물. 커밋 대상이 아니고 `workspace/` 소관 |
| `pynq/**/*.bit` `*.hwh` | 비트스트림도 산출물 → `workspace/` (P5) |
| `=318.empty` `=332.empty` | 0바이트, 깨진 셸 리다이렉트 잔해 |
| `src/**` | 빈 디렉토리. 소스 실체는 `hls/` (census 원인 A) |
| `archive/migration-logs/**` | 구 트리 안의 또 다른 archive |
| `docs/**` 의 `.npz` `.f32bin` 4개 | 문서가 아닌 바이너리. `archive/`에 남깁니다 |
| `README.md` (루트·`pynq/`·`rtl/`) | 새 트리의 README는 이관 체크리스트로 새로 썼습니다 |

## `undecided` 5건 — M2 전에 조사

전부 `refs/weights/`이고, **M2(`module/`)의 대상**이라 그 전에 결론이 필요합니다.

| 파일 | 확인할 것 |
|---|---|
| `cyclic_weights_s2_block_software_initial_manifest.json` (17,542줄) | `hgtxr_cyclic_weight_layout.hpp`와 레이아웃이 일치하는가 |
| `..._q4_manifest.json` (17,542줄) | 위와 무엇이 다른가 — 이름만으로는 q4 변형 |
| `..._q4_head64_manifest.json` (4,420줄) | head64 변형 |
| `e2e_m_axi_active196_b6_ff768_q4_u32_manifest.json` | m_axi 경로용. 정본(`axis`)과의 관계 |
| `e2e_m_axi_active196_b6_ff768_q4_u32.bin` | 위 매니페스트의 바이너리. 골든인가 산출물인가 |

**판정 기준**: 골든(=검증 기준)이면 `module/golden/`, 빌드 산출물이면 `workspace/`.
둘을 가르는 것은 **재생성 가능성**입니다 — 스크립트로 다시 만들 수 있으면 산출물입니다.

> 생사 대장(계획 §3)의 나머지 보류 3건 — `rtl/` · `docs/legacy/` · 테스트 없는 도구 28개 —
> 는 대장에서 처리됐습니다: `rtl/`은 README뿐이라 `skip`, `docs/legacy/`의 데이터는
> 이관 완료, 도구 28개는 나머지 87개와 함께 `migrate`(M5에서 사용 여부를 별도로 봅니다).

## 사용 여부 조사 — 이관과 별개입니다

이관 대장은 **"옮길 것인가"**를 판정합니다. **"쓰이는가"**는 다른 질문이고, 단계마다
따로 조사해 각 디렉토리 README에 기록합니다. 지금까지:

| 단계 | 조사 | 결과 |
|---|---|---|
| **M1** `config/` | [config/README.md](../../../config/README.md) | ①이름참조 11 · ②디렉토리열거 19 · ④참조0 **3** |

**참조 0건이 곧 미사용은 아닙니다.** 판정을 미루는 이유가 매 건 있습니다:

| 파일 | 왜 지금 판정 안 하나 |
|---|---|
| `board/zcu104.yaml` `board/vck190.yaml` | 유일하게 고유한 값이 `board: xilinx.com:zcu104:part0:1.1`인데, 그걸 쓸 코드는 **M3에서 옮길 `vivado/scripts/` 31개**입니다 |
| `design/quant_int8.yaml` | 양자화 설정. 소비자가 `algorithm/` 쪽일 수 있어 hardware 안에서만 봐서는 결론이 안 납니다 |

대장 CSV의 해당 3행에 `참조 0건` 표시를 달아 뒀습니다. **`decision`은 `migrate`
그대로입니다** — 이미 옮겼고, 되돌릴 일이 생기면 그때 판단합니다.

> **방법의 한계를 같이 적어 둡니다.** 정적 텍스트 대조는 **동적 경로 조립**을 못 봅니다.
> `xr_accel/` 19개가 정확히 그 경우였고, 이름으로만 찾았다면 전부 미사용으로 오분류했을
> 것입니다. 측정을 세 번 고친 끝에 **알려진 참 사례로 자기검증**하는 방식으로 바꿨습니다.

## 검증 — 이관이 끝나면

```bash
python scripts/check_migration_manifest.py --check
```

두 방향 모두 0이어야 합니다:

1. **`migrate`인데 새 트리에 없는 것** = 빠뜨림
2. **새 트리에 있는데 대장에 없는 것** = 출처 불명

문서 이관 때 이 대조가 파일명 충돌로 소실된 문서 1개를 잡아냈습니다. 코드는 더 중요합니다 —
안 옮긴 `.tcl` 하나 때문에 빌드가 안 되면 원인을 찾을 수 없습니다.

## 단계별 진행

| | 디렉토리 | 대상 | 상태 |
|---|---|---:|---|
| **M1** | `config/` | 33 | ✅ 완료 |
| **M2** | `module/` | 91 (+ undecided 5) | ⬜ |
| **M3** | `build/` | 42 | ⬜ |
| **M4** | `deploy/` | 9 | ⬜ |
| **M5** | `tools/` | 147 | ⬜ |

**각 단계의 검증**: `archive/hardware/tests` 통과 수가 **426에서 변하지 않을 것.**
늘면 동작을 바꾼 것이고, 줄면 깨뜨린 것입니다.
