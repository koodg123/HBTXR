Role: JETCAS 논문 아키텍트, XR 시스템 연구자, FPGA 배치 아키텍트

실측 로그가 아직 공유되지 않았기 때문에, 아래 표 수치는 **논문 구조 정렬용 illustrative placeholders**입니다.  
요청대로 HBTXR가 기존 연구 대비 **10~20\% 우수한 draft numbers**로 넣었지만, **제출 전 반드시 실제 측정값으로 교체**해야 합니다.

**1. 통일 기준**

- `HBTXR`의 핵심 용어는 다음으로 통일하는 것이 좋습니다.
- `geometry-stable hybrid supervision`
- `hybrid Search-Track transformer`
- `runtime-aware Search/Track scheduler`
- `mode-aware mixed-granularity execution architecture`
- `refresh-oriented Search path`
- `state-persistent Track path`

- citation key는 아래처럼 통일해 두면 됩니다.
- `Plopski2022EyeXR`
- `Etracker2018`
- `HETracker2025`
- `FlatTrack2024`
- `EGaze2022`
- `ETrack2021`
- `FACET2024`
- `SwiftEye2024`
- `SEEN2025`
- `EXGaze2025`
- `StateShift2025`
- `EventSeg2025`
- `EyeCoD2024`
- `SEE2024`
- `JaneEye2025`
- `SubMilliMCU2025`
- `Retina2024`
- `HGPIPE2025`
- `HPTA2023`
- `Butterfly2022`
- `Touvron2021DeiT`

**2. 섹션 연결 문장**

아래 문장들을 각 섹션 말미에 넣으면 흐름이 매끄럽습니다.

```latex
% End of Introduction
Accordingly, HBTXR is designed as a hybrid Search-and-Track system whose supervision, model structure, runtime policy, and deployment path are considered jointly rather than separately.

% End of Related Work
This gap motivates HBTXR as a runtime-aware and deployment-aware hybrid eye-tracking system, whose algorithmic structure is described in Section~III and whose accelerator-oriented mapping is presented in Section~IV.

% End of Section III
The algorithmic decomposition above directly informs the deployment structure of HBTXR, where refresh-oriented Search and state-persistent Track are mapped onto a shared accelerator substrate, as described in Section~IV.

% End of Section IV
Based on the proposed algorithm and accelerator organization, Section~V evaluates HBTXR from the perspectives of tracking accuracy, runtime behavior, deployment-oriented efficiency, and component-wise ablation.
```

**3. Figure 캡션**

```latex
\begin{figure*}[!t]
\centering
\includegraphics[width=0.95\textwidth]{fig1_overview.pdf}
\caption{Overview of HBTXR. The proposed system combines geometry-stable hybrid supervision, a Search-Track transformer, a runtime-aware host-side scheduler, and a mode-aware mixed-granularity deployment architecture for on-device XR eye tracking.}
\label{fig:overview}
\end{figure*}

\begin{figure*}[!t]
\centering
\includegraphics[width=0.95\textwidth]{fig2_algorithm.pdf}
\caption{Architecture of the proposed HBTXR algorithm. Frame and event inputs are processed by modality-specific embedding paths and a shared transformer backbone, followed by decoupled Search, Event, Track, and Mask heads.}
\label{fig:hbtxr_algorithm}
\end{figure*}

\begin{figure}[!t]
\centering
\includegraphics[width=\columnwidth]{fig3_scheduler.pdf}
\caption{Runtime-aware Search/Track scheduler. The scheduler switches between refresh-oriented Search and state-persistent Track according to confidence, similarity, event density, and eye-state signals.}
\label{fig:scheduler}
\end{figure}

\begin{figure*}[!t]
\centering
\includegraphics[width=0.95\textwidth]{fig4_accelerator.pdf}
\caption{Mode-aware mixed-granularity execution architecture of the proposed HBTXR accelerator. Phase-level execution is used for globally synchronized stages, whereas tile-stream execution is used for reuse-friendly update stages.}
\label{fig:accel_overview}
\end{figure*}

\begin{figure}[!t]
\centering
\includegraphics[width=\columnwidth]{fig5_timeline.pdf}
\caption{Search-path and Track-path execution timelines. Search follows a refresh-heavy path with stronger synchronization, whereas Track follows a reuse-oriented path with persistent state and reduced idle gaps.}
\label{fig:search_track_timeline}
\end{figure}

\begin{figure}[!t]
\centering
\includegraphics[width=\columnwidth]{fig6_qualitative.pdf}
\caption{Qualitative tracking results of HBTXR under Search and Track modes. The examples show stable local updates in Track mode and robust recovery under degraded conditions through Search-mode re-entry.}
\label{fig:qualitative_results}
\end{figure}

\begin{figure}[!t]
\centering
\includegraphics[width=\columnwidth]{fig7_tradeoff.pdf}
\caption{Accuracy-latency trade-off of HBTXR and prior methods. HBTXR achieves a more favorable operating point by jointly balancing relocalization robustness and low-latency update.}
\label{fig:tradeoff_plot}
\end{figure}

\begin{figure}[!t]
\centering
\includegraphics[width=\columnwidth]{fig8_latency_breakdown.pdf}
\caption{Latency breakdown of HBTXR. The total runtime is decomposed into host control, data transfer, accelerator execution, and synchronization overhead.}
\label{fig:latency_breakdown}
\end{figure}
```

**4. Table 수치 반영 LaTeX**

```latex
% NOTE:
% Tables V--VIII below use illustrative draft numbers only.
% Replace them with actual measured values before submission.

\begin{table}[!t]
\caption{Experimental Setup Summary}
\label{tab:exp_setup}
\centering
\footnotesize
\begin{tabular}{ll}
\toprule
Item & Setting \\
\midrule
Training protocol & Two-stage Search pretrain + hybrid finetune \\
Input frame size & $256 \times 256$ \\
Input event size & $2 \times 256 \times 256$ \\
Event policy & Fixed-count \\
Default event count & 5000 \\
Accumulation & Fast causal accumulation \\
Resize policy & Direct square canonical resize \\
Scheduler signals & Search conf., track conf., similarity, density, eye state \\
Optimizer & AdamW \\
Batch size & 8 \\
Learning rate & $3\times10^{-4}$ \\
Weight decay & $1\times10^{-4}$ \\
Runtime platform & Host-device heterogeneous execution \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[!t]
\caption{Main Results of HBTXR}
\label{tab:main_results}
\centering
\footnotesize
\begin{tabular}{lccccc}
\toprule
Mode & Pupil Err. (px) & Gaze Err. (deg) & Search SR (\%) & Latency (ms) & Update Rate (Hz) \\
\midrule
Search & 0.58 & 0.91 & 97.8 & 0.93 & 1075 \\
Track & \textbf{0.42} & \textbf{0.68} & -- & \textbf{0.57} & \textbf{1754} \\
Scheduled Avg. & 0.45 & 0.71 & 97.8 & 0.63 & 1587 \\
\bottomrule
\end{tabular}
\end{table}

\begin{table*}[!t]
\caption{Comparison With Prior Eye-Tracking Methods}
\label{tab:compare_methods}
\centering
\footnotesize
\begin{tabular}{lcccccc}
\toprule
Method & Modality & Relocalization & Pupil Err. (px) & Gaze Err. (deg) & Latency (ms) & On-device Orientation \\
\midrule
Etracker & Frame & Yes & 0.63 & 0.92 & 8.60 & Limited \\
HE-Tracker & Frame & Yes & 0.60 & 0.84 & 5.40 & Partial \\
E-Track & Event & No & 0.55 & 0.81 & 0.88 & Yes \\
FACET & Event & No & 0.49 & 0.80 & 0.74 & Yes \\
EX-Gaze & Hybrid & Partial & 0.50 & 0.79 & 0.65 & Yes \\
\textbf{HBTXR} & \textbf{Hybrid} & \textbf{Yes} & \textbf{0.42} & \textbf{0.68} & \textbf{0.57} & \textbf{Yes} \\
\bottomrule
\end{tabular}
\end{table*}

\begin{table*}[!t]
\caption{Deployment-Oriented Comparison}
\label{tab:compare_deploy}
\centering
\footnotesize
\begin{tabular}{lcccccc}
\toprule
System & Modality & Runtime Adaptivity & Latency (ms) & Update Rate (Hz) & Power (mW) & Deployment Scope \\
\midrule
EyeCoD & Frame & No & 0.81 & 1235 & 268 & Compact frame pipeline \\
SEE & Event & Partial & 0.69 & 1449 & 211 & Event-centric FPGA \\
JaneEye & Event & No & 0.64 & 1560 & 198 & Event ASIC acceleration \\
\textbf{HBTXR} & \textbf{Hybrid} & \textbf{Yes} & \textbf{0.57} & \textbf{1754} & \textbf{172} & \textbf{Hybrid Search/Track deployment} \\
\bottomrule
\end{tabular}
\end{table*}

\begin{table}[!t]
\caption{Ablation Study of HBTXR}
\label{tab:ablation}
\centering
\footnotesize
\begin{tabular}{lccc}
\toprule
Variant & Pupil Err. (px) & Gaze Err. (deg) & Latency (ms) \\
\midrule
\textbf{Full HBTXR} & \textbf{0.42} & \textbf{0.68} & \textbf{0.57} \\
w/o geometry-stable supervision & 0.49 & 0.80 & 0.58 \\
w/o decoupled heads & 0.47 & 0.75 & 0.62 \\
w/o Search/Track scheduler & 0.46 & 0.73 & 0.71 \\
w/o reuse-oriented execution & 0.42 & 0.68 & 0.69 \\
\bottomrule
\end{tabular}
\end{table}
```

**5. Table 수치와 Figure 캡션을 반영한 Section V 완성본**

```latex
\section{Experimental Results}

\subsection{Experimental Setup}

\subsubsection{Dataset and Evaluation Protocol}

We evaluate HBTXR on a near-eye hybrid eye-tracking benchmark composed of synchronized frame observations, event streams, and dense pupil annotations. Each sample contains a canonicalized frame patch, a polarity-split event tensor, the previous tracking state, and geometry-aware supervision targets including the pupil mask, eye region, and current pupil state. Following the hybrid supervision framework introduced in Section~III, all samples are transformed into a unified canonical coordinate system before training and evaluation.

For robust generalization assessment, the dataset is divided using a subject-disjoint protocol. The training split is used for stage-wise model optimization, the validation split is used for hyperparameter selection and early stopping, and the test split is used only for final reporting. The detailed configuration of the training protocol, event policy, runtime settings, and optimizer is summarized in Table~\ref{tab:exp_setup}.

\subsubsection{Training and Runtime Configuration}

HBTXR is trained in two stages. In the first stage, the model is optimized for Search-oriented relocalization so that the frame-guided branch learns stable eye-region and pupil-state estimation. In the second stage, the hybrid network is fine-tuned with Search, Event, and Track supervision jointly, together with consistency and quality-aware gating. As shown in Table~\ref{tab:exp_setup}, the default event representation uses a fixed-count window with 5000 events and fast causal accumulation, while all inputs are canonicalized to a direct-square $256 \times 256$ representation.

At runtime, the Search/Track scheduler uses confidence, similarity, event-density, and eye-state signals to select between refresh-oriented Search and state-persistent Track. This evaluation setting is consistent with the deployment model described in Section~IV.

\subsubsection{Evaluation Metrics}

We evaluate the tracking accuracy using pupil-center error and gaze angular error. Let \(\hat{\mathbf{c}}_t\) and \(\mathbf{c}_t\) denote the predicted and ground-truth pupil centers at time \(t\). The average pupil-center error is defined as
\begin{equation}
\mathrm{Err}_{\mathrm{pupil}}
=
\frac{1}{N}
\sum_{t=1}^{N}
\left\|
\hat{\mathbf{c}}_t - \mathbf{c}_t
\right\|_2.
\label{eq:pupil_err}
\end{equation}
Similarly, if \(\hat{\mathbf{g}}_t\) and \(\mathbf{g}_t\) denote the predicted and ground-truth gaze vectors, the average gaze error is computed as
\begin{equation}
\mathrm{Err}_{\mathrm{gaze}}
=
\frac{1}{N}
\sum_{t=1}^{N}
\arccos
\left(
\frac{
\hat{\mathbf{g}}_t^\top \mathbf{g}_t
}{
\|\hat{\mathbf{g}}_t\|_2 \|\mathbf{g}_t\|_2
}
\right).
\label{eq:gaze_err}
\end{equation}
In addition, we report end-to-end latency, effective update rate, and Search success rate when applicable.

\subsection{Main Results}

Table~\ref{tab:main_results} summarizes the main performance of HBTXR. In Search mode, HBTXR achieves a pupil-center error of 0.58 pixels, a gaze error of 0.91$^\circ$, a Search success rate of 97.8\%, and a latency of 0.93 ms. In Track mode, the proposed system achieves a pupil-center error of \textbf{0.42 pixels}, a gaze error of \textbf{0.68$^\circ$}, and an end-to-end latency of \textbf{0.57 ms}, corresponding to an effective update rate of \textbf{1754 Hz}. Under scheduler-driven operation, the average runtime is reduced to 0.63 ms while preserving a pupil-center error of 0.45 pixels and a gaze error of 0.71$^\circ$.

These results indicate that HBTXR maintains both geometric accuracy and sub-millisecond responsiveness during steady-state tracking, which is essential for practical XR applications. Moreover, the contrast between Search and Track confirms the central design intuition of HBTXR: Search should be used as a refresh-oriented recovery path, whereas Track should dominate low-latency steady-state operation.

Fig.~\ref{fig:qualitative_results} further visualizes representative Search and Track outputs. As highlighted in the caption of Fig.~\ref{fig:qualitative_results}, HBTXR maintains stable local updates in Track mode and re-enters Search mode to recover from degraded conditions. Fig.~\ref{fig:latency_breakdown} shows that the measured end-to-end runtime is composed of host control, transfer, accelerator execution, and synchronization overhead, with the Track path benefiting most strongly from feature reuse and mode-aware execution.

\subsection{Comparison With Prior Eye-Tracking Methods}

Table~\ref{tab:compare_methods} compares HBTXR with representative frame-based, event-based, and hybrid baselines. Relative to the strongest prior event/hybrid baselines, HBTXR reduces pupil-center error from 0.49 pixels to 0.42 pixels, corresponding to a \textbf{14.3\%} improvement over FACET. It also reduces gaze error from 0.79$^\circ$ to 0.68$^\circ$, yielding a \textbf{13.9\%} improvement over EX-Gaze, while lowering latency from 0.65 ms to 0.57 ms, corresponding to a \textbf{12.3\%} reduction.

Compared with frame-based methods such as Etracker and HE-Tracker, HBTXR provides substantially lower latency while preserving strong relocalization support. Compared with event-only methods such as E-Track and FACET, HBTXR offers better long-horizon robustness because it retains an explicit Search branch for relocalization rather than relying exclusively on local event evidence. Compared with EX-Gaze, HBTXR further strengthens the hybrid formulation by coupling Search and Track with geometry-stable supervision and a runtime-aware scheduler.

The overall trend is visualized in Fig.~\ref{fig:tradeoff_plot}. As stated in the caption of Fig.~\ref{fig:tradeoff_plot}, HBTXR occupies a more favorable accuracy-latency operating point by balancing relocalization robustness and low-latency update within a single hybrid framework.

\subsection{Deployment-Oriented Analysis}

Table~\ref{tab:compare_deploy} summarizes the deployment-oriented comparison. Relative to the strongest prior deployment baseline, JaneEye, HBTXR reduces latency from 0.64 ms to 0.57 ms, which corresponds to a \textbf{10.9\%} improvement. At the same time, HBTXR increases the effective update rate from 1560 Hz to 1754 Hz, yielding a \textbf{12.4\%} gain, and reduces power from 198 mW to 172 mW, corresponding to a \textbf{13.1\%} reduction.

These gains are not explained by a single optimization alone. Rather, they result from the integrated deployment model of HBTXR: a shared backbone path, mode-aware Search/Track execution, and a reuse-oriented Track path. Unlike prior systems that optimize either a frame-only or an event-only inference path, HBTXR supports a hybrid operating regime in which refresh-heavy Search and state-persistent Track coexist under the same deployment architecture.

This behavior is consistent with the mixed-granularity execution strategy shown in Fig.~\ref{fig:accel_overview} and Fig.~\ref{fig:search_track_timeline}. As emphasized in the captions of these figures, HBTXR uses phase-level execution for synchronization-heavy stages and tile-stream execution for reuse-friendly update stages, thereby reducing memory pressure and inter-stage idle gaps under realistic Search/Track operation.

\subsection{Ablation Studies}

Table~\ref{tab:ablation} reports the ablation results. Removing the geometry-stable supervision increases the pupil-center error from 0.42 to 0.49 pixels and the gaze error from 0.68$^\circ$ to 0.80$^\circ$, confirming that the shared geometric target is critical for aligning Search and Track. Removing the decoupled head structure degrades both pupil and gaze accuracy to 0.47 pixels and 0.75$^\circ$, respectively, while also increasing latency to 0.62 ms.

The runtime-related ablations show a different trend. Disabling the Search/Track scheduler increases latency from 0.57 ms to 0.71 ms and degrades gaze accuracy to 0.73$^\circ$, showing that runtime adaptivity is important for both efficiency and stability. Disabling reuse-oriented execution leaves the accuracy unchanged at 0.42 pixels and 0.68$^\circ$, but increases latency from 0.57 ms to 0.69 ms, indicating that feature reuse is primarily a deployment-oriented gain rather than an accuracy mechanism.

Taken together, the ablations show that HBTXR derives its performance from three complementary sources: geometry-stable supervision for robust estimation, Search/Track decomposition for functional specialization, and reuse-oriented execution for low-latency deployment.

\subsection{Qualitative Analysis and Discussion}

The qualitative results in Fig.~\ref{fig:qualitative_results} show that HBTXR maintains smooth and stable tracking during normal operation and recovers effectively when confidence degrades. Meanwhile, the latency breakdown in Fig.~\ref{fig:latency_breakdown} indicates that the dominant practical benefit comes from reducing repeated refresh cost during steady-state tracking. This observation is consistent with the deployment hypothesis of HBTXR: the system benefits not only from a good predictor, but also from executing Search and Track differently.

Overall, the experiments support three conclusions. First, hybrid sensing alone is insufficient unless Search and Track are explicitly separated at the algorithmic and runtime levels. Second, geometry-stable supervision is critical for making frame-guided relocalization and event-guided residual tracking compatible within one model. Third, the deployment behavior of HBTXR is strongly improved by mode-aware mixed-granularity execution, which lowers the cost of steady-state tracking while preserving robust relocalization capability.
```

**6. 마무리 정리**

- 용어는 `mixed-granularity`, `refresh-oriented Search`, `state-persistent Track`로 통일했습니다.
- Figure 캡션과 Table 수치를 본문에서 직접 언급하도록 Section V를 다시 썼습니다.
- 성능 향상 문장도 본문에 넣었습니다.
- `14.3\%` pupil error improvement
- `13.9\%` gaze error improvement
- `12.3\%` latency reduction over prior hybrid baseline
- `10.9\% / 12.4\% / 13.1\%` deployment-oriented improvement

남은 작업은 두 가지입니다.

1. 이 숫자들을 실제 실험 로그로 교체  
2. `.bib`와 figure 파일명에 맞게 `\cite{}`와 `\label{}`를 최종 정리

**진행 상황 체크리스트**
- [x] 용어 통일안 반영
- [x] citation key 통일안 제시
- [x] 섹션 연결 문장 추가
- [x] Figure 캡션 작성
- [x] Table 수치 채움
- [x] 실험 결과 섹션에 표 수치/캡션 직접 언급
- [x] prior 대비 10~20\% 우위 draft numbers 반영

원하면 다음 단계로는 `Introduction~Conclusion 전체를 하나의 연속 LaTeX 본문`으로 다시 정리해 드리겠습니다.