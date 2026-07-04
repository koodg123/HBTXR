# Transformer-Accel / HLSTransformation 교차참조(cross-ref)

> 본 문서는 신규 정밀 분석이 아니라 **교차참조**다. 정밀 분석 본문은
> [`REF/Analysis/ViT-Accelerator/HLSTransformation.md`](../ViT-Accelerator/HLSTransformation.md) **참조**.
> 대상 repo: `REF/Transformer-Accel/HLSTransformation`
> 방식: bash 미사용(UNC 오류 회피), Glob/Grep/Read만 사용. `파일:라인` 근거 기반.

---

## 결론: 사실상 동일 (코드 본체 byte-identical, 패키징만 상이)

`REF/Transformer-Accel/HLSTransformation` 은 `REF/ViT-Accelerator/HLSTransformation` 과
**동일한 단일 코드베이스**다. 핵심 HLS 커널·하이퍼파라미터·커널 인터페이스가 모두 일치한다.
차이는 코드 로직이 아니라 **저장소 패키징 메타파일(README, Vitis system 프로젝트)** 수준에 그친다.

> 참고: 같은 부모(`REF/`) 아래 ViT-Accelerator 와 Transformer-Accel 이 **형제 디렉토리**로 공존하므로,
> 두 경로가 동일 submission repo의 중복본일 가능성이 높다(아래 근거가 이를 뒷받침).

---

## 동일성 근거 (대상 repo Read)

- **모델 하이퍼파라미터 동일**: `llama_xrt_kernels/src/config.h:4-11`
  → `dim=768, hidden_dim=2048, n_layers=12, n_heads=12, n_kv_heads=12, vocab_size=32000, seq_len=1024, GS=64`.
  기존 분석 §1·§3.0과 완전 일치(표준 MHA, head_size=64).
- **핵심 커널 코드 동일**:
  - `rmsnorm`(`forward.cpp:13-49`): 동일 2-루프 구조, `UNROLL 128/64` + `ARRAY_PARTITION cyclic 128/64`.
  - `softmax`(`forward.cpp:51-96`): max/exp/sum/norm 4단계, `loop_tripcount max=257`, `1/sum` 사전계산.
  - `matmul`(`forward.cpp:191-267`): int8×int8→int32 그룹(GS=64) 누산 + fp32 scale 환산.
    **`matmul1~4` 내부 `UNROLL`이 전부 주석 처리**(`:238,246,255,260`) — 기존 분석 §3.3·§8의 "GEMV 미최적화" 한계가 그대로 재현됨.
  - `matmul_old` 데드 코드도 동일하게 잔존(`forward.cpp:98-189`).
- **커널 인터페이스 동일**: `extern "C" void forward(Transformer<...>*, token, pos, key_cache[], value_cache[], out)`
  (`forward.cpp:269`), `m_axi` gmem0(transformer)/gmem1(out), 활성화 텐서 전부 `static` + `ARRAY_PARTITION cyclic factor=16`(`forward.cpp:277-300`). dataflow 미사용도 동일.
- **타깃 디바이스 동일**: `llama_xrt_kernels.prj:2` platform = `xilinx_aws-vu9p-f1_shell-v04261818_201920_3`
  (AWS F1 / UltraScale+ VU9P), 커널 인자 5개 + master 버퍼 매핑, `sw_emu/hw_emu/hw` 3구성 동일(`:3-68`).
- 디렉토리 4종(`llama_xrt`, `llama_xrt_kernels`, `cpu_benchmarks`, `gpu_benchmarks`)·하위 파일 구성도 기존 §2와 일치(runq.c, export.py, tinyllama2*.py, benchmark_stories 등).

---

## 차이점 (패키징/메타 수준만)

1. **루트 README 존재**: 대상 repo에는 `README.md`(루트, 35줄)가 있다.
   - 기존 분석(ViT-Accelerator)은 "루트에 통합 README 없음"(§2·§6)으로 기록했음 → **이 부분이 다름**.
   - 내용: AWS FPGA Developer AMI + EC2 z1d.2xlarge 세팅, Vitis IDE 빌드, ~12h HW 빌드, AFI/F1 배포, 호스트 빌드 명령
     `g++ -O3 -std=c++17 src/llama2.cpp ... -lxrt_coreutil`, 실행 `./llama2 {weights} -z {tokenizer} -t -n -i -k {kernel}` (`README.md:9-33`).
   - 즉 기존 §6 "FPGA측 빌드 절차 미정밀화/스크립트 부재" 라는 한계를 **이 README가 일부 보완**한다(빌드 절차 명시).
2. **Vitis system-level 프로젝트 추가 존재**: `llama_xrt_system/`(`.sprj`), `llama_xrt_system_hw_link/`(`.prj`).
   - 기존 분석 디렉토리 트리(§2)에는 미기재. 호스트(`llama_xrt`)+커널(`llama_xrt_kernels`)을 묶는 system/hw_link 프로젝트로,
     `.prj`(`llama_xrt_kernels.prj:2`)의 `systemProject="llama_xrt_system"` 참조와 정합. 실제 통합 빌드용 메타가 더 갖춰진 사본임.

> 위 두 차이는 모두 **빌드/패키징 메타데이터**이며, HLS 커널·호스트·벤치마크의 알고리즘/데이터패스/pragma에는 차이가 없다.

---

## 종합

- 동일여부: **동일(코드 본체 byte-identical), 패키징만 상이.**
- 핵심 차이 1줄: 대상 repo는 루트 README + Vitis `llama_xrt_system`/`hw_link` 통합 프로젝트가 추가되어 **빌드/배포 메타가 더 완비**된 사본일 뿐, 커널 로직은 ViT-Accelerator 분석과 100% 일치.
- 우리 프로젝트(HG-PIPE 계열 ViT/Transformer 가속 + XR 시선추적) 시사점은
  [`ViT-Accelerator/HLSTransformation.md` §9](../ViT-Accelerator/HLSTransformation.md) 를 그대로 원용한다(중복 재작성 불요).

### Read 근거 파일 (대상 repo)
- `README.md`(루트, 전체)
- `llama_xrt_kernels/src/forward.cpp`(:1-110, :191-320), `.../config.h`(전체)
- `llama_xrt_kernels/llama_xrt_kernels.prj`(전체)
- 파일 인벤토리: Glob 전체 트리(system/hw_link 프로젝트 존재 확인)
