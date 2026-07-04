"""
05_eval_annotation_quality.py — synthesize a LABEL-QUALITY report from the
outputs of scripts 01-04, mapped to the EV-Eye benchmark axes + the 4 label
diagnostics (Q1 consistency, Q2 label noise, Q3 coverage, Q4 sync).

It reads <out>/01_*.json, 02_*.{json,csv}, 03_*.{json,csv}, 04_*.json and writes:
  <out>/annotation_quality_report.md
  <out>/plots/coverage_by_session.png
  <out>/plots/label_jitter_hist.png   (if matplotlib available)

Usage:
  python 05_eval_annotation_quality.py --out ../results
"""
import os, csv, json, argparse, statistics
import evlib as ev


def load_json(path, default=None):
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default


def load_csv(path):
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../results")
    args = ap.parse_args()
    O = args.out

    s01 = load_json(ev.p(O, "01_summary.json"), {})
    s02 = load_json(ev.p(O, "02_labels_summary.json"), {})
    via = load_csv(ev.p(O, "02_via_coverage.csv"))
    s03 = load_json(ev.p(O, "03_tobii_summary.json"), {})
    sync = load_csv(ev.p(O, "03_tobii_sync.csv"))
    s04 = load_json(ev.p(O, "04_track_summary.json"), {})

    # ---- Q2 label-noise proxy: distribution of per-session ellipse-center jitter
    jit = [fnum(r.get("center_jitter_px")) for r in via]
    jit = [j for j in jit if j is not None]
    # ---- Q3 coverage by session
    cov_by_sess = {}
    for r in via:
        s = r.get("session"); lab = fnum(r.get("n_labelled")) or 0
        cov_by_sess[s] = cov_by_sess.get(s, 0) + lab
    # ---- Q4 sync offsets
    offs = [fnum(r.get("ttl_minus_send0")) for r in sync]
    offs = [o for o in offs if o is not None]

    # ---- plots
    plotdir = ev.ensure_dir(ev.p(O, "plots"))
    plotted = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if cov_by_sess:
            plt.figure(figsize=(6, 3.2))
            ks = [k for k in ev.SESSIONS if k in cov_by_sess]
            plt.bar(ks, [cov_by_sess[k] for k in ks], color="#4C78A8")
            plt.title("VIA labelled frames by session"); plt.ylabel("labelled frames")
            plt.xticks(rotation=20, ha="right"); plt.tight_layout()
            fp = ev.p(plotdir, "coverage_by_session.png"); plt.savefig(fp, dpi=120); plt.close()
            plotted.append(fp)
        if jit:
            plt.figure(figsize=(6, 3.2))
            plt.hist(jit, bins=20, color="#E45756")
            plt.title("Per-session ellipse-center jitter (label-noise proxy)")
            plt.xlabel("mean successive center delta (px)"); plt.ylabel("sessions")
            plt.tight_layout()
            fp = ev.p(plotdir, "label_jitter_hist.png"); plt.savefig(fp, dpi=120); plt.close()
            plotted.append(fp)
    except Exception as e:
        plotted.append(f"(matplotlib unavailable: {e})")

    # ---- report
    def stat(a):
        if not a:
            return "n/a"
        return (f"n={len(a)}, min={min(a):.3f}, median={statistics.median(a):.3f}, "
                f"mean={statistics.mean(a):.3f}, max={max(a):.3f}")

    L = []
    L.append("# EV-Eye 라벨 품질 평가 리포트 (자동 생성)\n")
    L.append("> `05_eval_annotation_quality.py`가 01–04 산출물을 종합한 결과입니다.\n")

    L.append("## 0. 데이터셋 규모")
    t = (s01 or {}).get("totals", {})
    L.append(f"- users={s01.get('users','?')}, sessions={t.get('sessions','?')}, "
             f"frames={t.get('frames','?'):,}" if isinstance(t.get('frames'), int)
             else f"- users={s01.get('users','?')}, sessions={t.get('sessions','?')}")
    L.append(f"- VIA 라벨 프레임(총)={t.get('labelled_frames','?')}, "
             f"events(est)={t.get('events_est','?'):,}" if isinstance(t.get('events_est'), int)
             else f"- VIA 라벨 프레임(총)={t.get('labelled_frames','?')}")
    pres = (s01 or {}).get("processed_data_present", {})
    if pres:
        L.append("- processed_data 폴더 존재: " +
                 ", ".join(f"{k}={'O' if v else 'X'}" for k, v in pres.items()))
    L.append("")

    L.append("## 1. Q3 — 라벨 커버리지 / 세션 균형")
    if cov_by_sess:
        for k in ev.SESSIONS:
            if k in cov_by_sess:
                L.append(f"- {k} ({ev.SESSION_MEANING.get(k,'?')}): {int(cov_by_sess[k])} labelled")
    mbs = (s02 or {}).get("mask_files_by_session", {})
    L.append(f"- mask .h5 by session: {mbs}")
    if mbs.get("session_1_0_1", 0) == 0:
        L.append("- ⚠ **session_1_0_1 마스크 라벨 없음** 확인 (EV-Eye 사양과 일치).")
    L.append("")

    L.append("## 2. Q2 — 라벨 노이즈 프록시 (타원중심 지터)")
    L.append(f"- 세션별 평균 인접-중심 변화(px): {stat(jit)}")
    L.append("- 해석: 값이 클수록 라벨 흔들림/실제 동공운동 혼재. **이벤트/프레임 PE가 이 "
             "지터보다 낮으면** 라벨 노이즈 하한을 의심(과적합/누설).")
    L.append("")

    L.append("## 3. Q1 — 마스크↔타원 일관성")
    if os.path.isfile(ev.p(O, "02_mask_stats.csv")):
        L.append("- `02_mask_stats.csv`에 마스크 면적·중심·적합타원 산출됨. "
                 "VIA 타원중심과 동일 프레임 비교로 IoU/중심거리 확정 가능"
                 "(정렬 스키마 확인 후). 마스크가 타원의 결정론적 함수이므로 IoU≈1 기대; "
                 "이탈 시 `generate_pupil_mask.m` 재현 문제 신호.")
    else:
        L.append("- 마스크 기하 미산출(h5py 미설치 또는 마스크 폴더 부재).")
    L.append("")

    L.append("## 4. Q4 — 장치간 클럭 동기 (DAVIS µs UNIX ↔ Tobii 상대초)")
    L.append(f"- TTL offset(ttl0−send0) 분포: {stat(offs)}")
    L.append("- 해석: 사용자간 offset 분산이 크면 시선 DoD 평가의 정렬 오차원. "
             "Tobii gaze2d는 정규화[0,1] 장면좌표, timestamp는 녹화상대초.")
    L.append("")

    L.append("## 5. 추적 결과(.mat) 구조")
    if s04:
        L.append(f"- tracking files={s04.get('tracking_files','?')}, "
                 f"coverage={json.dumps(s04.get('coverage',{}).get('by_session',{}), ensure_ascii=False)}")
        if "trajectory_length" in s04:
            L.append(f"- 궤적 길이: {s04['trajectory_length']}")
    else:
        L.append("- 04 산출물 없음(scipy/h5py 미설치 또는 폴더 부재).")
    L.append("")

    L.append("## 6. EV-Eye 4대 벤치마크 매핑")
    L.append("| 지표 | 산출 위치 | 본 툴킷 연계 |")
    L.append("|---|---|---|")
    L.append("| ① IoU/F1 (분할) | predict vs mask.h5 | 02 마스크기하 → 별도 분할평가 |")
    L.append("| ② 프레임 PE | 분할중심 vs VIA중심 | 02 + 05 Q1 |")
    L.append("| ③ 이벤트 PE | .mat 추적중심 vs GT | 04 궤적 → PE 계산 |")
    L.append("| ④ 시선 DoD | 추적→다항회귀 vs Tobii | 03 + 04 |")
    L.append("")

    L.append("## 7. 그림")
    for fp in plotted:
        if fp.endswith(".png"):
            L.append(f"- `{os.path.relpath(fp, O)}`")
        else:
            L.append(f"- {fp}")
    L.append("")

    L.append("## 8. 자동 판정 요약")
    flags = []
    if mbs.get("session_1_0_1", 0) == 0:
        flags.append("session_1_0_1 마스크 결손(설계상 정상이나 평가셋 구성시 제외 필요)")
    if jit and statistics.median(jit) > 5:
        flags.append(f"라벨 중심 지터 중앙값 {statistics.median(jit):.2f}px — 라벨/운동 변동 큼")
    if offs and (max(offs) - min(offs)) > 0.05:
        flags.append("TTL offset 사용자간 편차 큼 — 시선동기 점검 권장")
    if not flags:
        flags.append("자동 임계 위반 없음(데이터 적재 여부 확인).")
    for fl in flags:
        L.append(f"- {fl}")

    report = ev.p(O, "annotation_quality_report.md")
    with open(report, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"[i] wrote {report}")
    for fp in plotted:
        print(f"    plot: {fp}")


if __name__ == "__main__":
    main()
