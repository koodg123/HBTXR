**1. Table I–VIII 구성안**

| Table | Caption | Recommended Columns |
|---|---|---|
| Table I | **Comparison of Prior Eye-Tracking Systems for XR and On-Device Deployment** | Work, Category, Modality, Primary Task, Search/Track Separation, Geometry Target, End-to-End Training, Hardware-Aware Design, XR/On-Device Target, Reported Latency |
| Table II | **Implementation Summary of the Proposed HBTXR System** | Item, Value |
| Table III | **Datasets, Splits, and Evaluation Protocol** | Dataset/Split, Subjects, Modality, Sensor Resolution, Annotation Type, Number of Samples/Sequences, Usage |
| Table IV | **Baseline Methods Used in Experimental Comparison** | Method, Category, Modality, Output Task, Runtime Platform, Key Characteristic, Source |
| Table V | **Main Performance of HBTXR Under Search and Track Modes** | Mode, Pupil-Center Error (px), Gaze Error (deg), Search Success Rate (%), End-to-End Latency (ms), Effective Update Rate (Hz), Notes |
| Table VI | **Comparison With Prior Eye-Tracking Algorithms** | Method, Category, Modality, Pupil Error (px), Gaze Error (deg), Latency (ms), Re-localization Support, XR/On-Device, End-to-End |
| Table VII | **Comparison With Prior Eye-Tracking Accelerators and Deployment-Oriented Systems** | Work, Platform, Modality, Backbone Type, Precision, Latency, Throughput/Update Rate, Power/Energy, Runtime Scheduler, Notes |
| Table VIII | **Ablation Study of HBTXR** | Variant, Canonical Contract, Hybrid Input, Decoupled Heads, Search/Track Scheduler, Feature Reuse, Pupil Error (px), Gaze Error (deg), Latency (ms) |

`Table II`의 `Item` 예시는 다음 정도면 충분하다: Host CPU, FPGA board/device, clock frequency, numeric precision, on-chip buffer size, DMA interface, backbone size, token size, operating modes.

---

**2. Fig. 1–12 캡션 초안**

| Figure | Caption |
|---|---|
| Fig. 1 | **Overview of HBTXR.** The proposed system integrates dense-annotation-aware canonicalization, dual frame/event sensing, a shared Partial DeiT-Tiny backbone, decoupled Search/Track/Mask heads, host-side runtime control, and CPU-FPGA workload partitioning for on-device XR eye tracking. |
| Fig. 2 | **Architecture of the proposed hybrid frame-event eye-tracking transformer.** Frame and event inputs are tokenized, fused through a shared transformer encoder, and processed by decoupled Search, Track, and Mask heads. |
| Fig. 3 | **State transition diagram of the two-stage Search/Track scheduler.** The host-side controller switches between global relocalization and low-latency local tracking based on confidence and geometric validity. |
| Fig. 4 | **Training pipeline of HBTXR.** Dense annotations are used to canonicalize heterogeneous inputs and to supervise search, tracking, masking, and geometry-consistency objectives in an end-to-end manner. |
| Fig. 5 | **System-level CPU-FPGA architecture of HBTXR.** The CPU handles sensing and runtime control, while the FPGA executes tokenization, transformer inference, head computation, and mode-aware refinement. |
| Fig. 6 | **Microarchitecture of the shared transformer engine.** The engine supports multimodal token processing through pipelined projection, attention, aggregation, and feed-forward stages. |
| Fig. 7 | **Streaming-based feature reuse and on-chip dataflow in Track mode.** Cached frame-side features are reused while only event-dependent updates are recomputed to reduce latency and memory traffic. |
| Fig. 8 | **FSM-based runtime scheduling on FPGA.** The accelerator-side FSM orchestrates Search and Track execution paths, buffer reuse, and head invocation with minimal host intervention. |
| Fig. 9 | **Qualitative results of HBTXR.** Example outputs in Search and Track modes under different gaze directions, motion patterns, and recovery conditions. |
| Fig. 10 | **Latency breakdown of HBTXR.** End-to-end runtime is decomposed into host preprocessing, transfer, FPGA inference, and host postprocessing for Search and Track modes. |
| Fig. 11 | **Accuracy-latency comparison with prior eye-tracking algorithms.** HBTXR achieves a favorable trade-off between pupil/gaze accuracy and low-latency operation. |
| Fig. 12 | **Comparison with prior accelerator and deployment-oriented systems.** HBTXR provides competitive latency while supporting hybrid sensing, transformer inference, and runtime-adaptive execution. |

실제로는 `Fig. 6`과 `Fig. 7`, 또는 `Fig. 7`과 `Fig. 8`을 합쳐서 그림 수를 줄여도 된다. IEEE 저널은 그림 수가 많으면 지면이 빠르게 늘어난다.

---

**3. IEEE LaTeX Skeleton**

```latex
\documentclass[journal]{IEEEtran}

\usepackage{cite}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{array}
\usepackage{algorithm}
\usepackage{algorithmic}
\usepackage{xcolor}

\begin{document}

\title{HBTXR: An Algorithm-Hardware Co-Designed Hybrid Eye Tracking System for On-Device XR Applications}

\author{Author~1, Author~2, Author~3%
\thanks{Manuscript received xxx; revised xxx.}
\thanks{Authors are with xxx. Corresponding author: xxx (e-mail: xxx).}
}

\maketitle

\begin{abstract}
Eye tracking is a key enabling technology for extended reality (XR), supporting gaze-based interaction, foveated rendering, and privacy-preserving on-device perception. However, existing approaches still suffer from a structural trade-off between semantic robustness and temporal responsiveness. Frame-based methods provide stable visual cues but incur high bandwidth cost and relatively large latency, whereas event-based methods offer high temporal resolution but remain limited in relocalization stability and representation consistency.

To address these limitations, this paper presents HBTXR, an algorithm-hardware co-designed hybrid eye tracking system that tightly integrates Frame Search and Event Track for on-device XR applications. HBTXR is built upon four key components: 1) a dense-annotation-aware canonical training contract, 2) a hybrid frame-event eye tracking transformer with decoupled search, track, and mask heads, 3) a two-stage host-side Search/Track scheduler, and 4) an FPGA accelerator architecture with a transformer engine, streaming-based feature reuse, and FSM-based runtime control. Experimental results show that HBTXR achieves a pupil-center error of 0.42 pixels, a gaze error of 0.68$^\circ$, and an end-to-end latency of 0.57 ms in tracking mode. These results demonstrate that HBTXR provides a practical solution for real-time on-device eye tracking in next-generation XR systems.
\end{abstract}

\begin{IEEEkeywords}
Eye tracking, extended reality, event camera, hybrid sensing, transformer accelerator, FPGA, on-device inference.
\end{IEEEkeywords}

% Optional only for local/non-IEEE version
% \section*{국문 초록}
% ...

\section{Introduction}
\subsection{Motivation and Background}
XR eye tracking is critical for gaze-based interaction, foveated rendering, and privacy-preserving on-device perception. Practical systems must satisfy both high accuracy and sub-millisecond responsiveness.

\subsection{Limitations of Existing Paradigms}
Frame-based methods are semantically stable but expensive in bandwidth and latency. Event-based methods are temporally efficient but weaker in relocalization and representation consistency. Existing hybrid systems remain fragmented across algorithm, scheduler, and hardware design.

\subsection{Problem Definition and Design Requirements}
A deployable hybrid eye-tracking system requires a canonical input contract, geometry-stable supervision, a real-time scheduler, and an accelerator-friendly backbone.

\subsection{Contributions}
The main contributions of this work are as follows:
\begin{itemize}
\item We propose a dense-annotation-aware canonical training contract.
\item We design an end-to-end hybrid frame-event eye-tracking transformer with decoupled heads.
\item We develop a two-stage Search/Track scheduler with host-side runtime control.
\item We present an FPGA-oriented algorithm-accelerator co-design for on-device XR deployment.
\end{itemize}

\subsection{Organization of the Paper}
The remainder of this paper is organized as follows. Section II reviews related work. Section III presents the proposed HBTXR algorithm. Section IV describes the accelerator architecture. Section V reports experimental results. Section VI concludes the paper.

\begin{figure*}[!t]
\centering
\includegraphics[width=0.95\textwidth]{fig1_overview.pdf}
\caption{Overview of HBTXR. The proposed system integrates dense-annotation-aware canonicalization, dual frame/event sensing, a shared Partial DeiT-Tiny backbone, decoupled Search/Track/Mask heads, host-side runtime control, and CPU-FPGA workload partitioning for on-device XR eye tracking.}
\label{fig:overview}
\end{figure*}

\section{Related Work}
\subsection{Frame-Based Eye Tracking}
Frame-based eye tracking uses RGB or IR images for pupil/gaze estimation and provides strong semantic stability, but suffers from high sensing and processing cost.

\subsection{Event-Based Eye Tracking}
Event-based methods exploit asynchronous sensing for low-latency updates, but often struggle with relocalization and representation consistency.

\subsection{Hybrid Event-Frame Eye Tracking}
Hybrid systems combine frame-based relocalization and event-based tracking, but many prior works do not jointly optimize scheduling and hardware realization.

\subsection{Eye Tracking Accelerator and Deployment-Oriented Methods}
Prior accelerators improve practical feasibility, but are often specialized either to frame-based CNNs or event-only sparse pipelines.

\subsection{Summary and Position of HBTXR}
HBTXR bridges the gap by combining hybrid sensing, end-to-end transformer inference, runtime scheduling, and FPGA-oriented co-design.

\begin{table*}[!t]
\caption{Comparison of Prior Eye-Tracking Systems for XR and On-Device Deployment}
\label{tab:prior_work}
\centering
\begin{tabular}{lccccccccc}
\toprule
Work & Category & Modality & Task & S/T Sep. & Geo. Target & E2E & HW-Aware & XR/Device & Latency \\
\midrule
Method A & Frame & Frame & Gaze & No & Partial & Yes & No & Partial & -- \\
Method B & Event & Event & Pupil & No & Yes & Yes & Yes & Yes & -- \\
Method C & Hybrid & Frame+Event & Pupil/Gaze & Partial & Partial & Partial & No & Yes & -- \\
HBTXR & Hybrid+Accel. & Frame+Event & Pupil/Gaze & Yes & Yes & Yes & Yes & Yes & 0.57 ms \\
\bottomrule
\end{tabular}
\end{table*}

\section{The Proposed HBTXR Algorithm}
\subsection{Problem Formulation and Hybrid Sensing}
Let $I_t \in \mathbb{R}^{H \times W}$ denote the frame image at time $t$, and let
\begin{equation}
\mathcal{E}_t = \{e_k = (x_k, y_k, p_k, \tau_k)\}_{k=1}^{N_t}
\label{eq:event_stream}
\end{equation}
denote the corresponding event stream.

We define a canonicalization transform as
\begin{equation}
\mathcal{T}_t: (I_t, \Psi(\mathcal{E}_t), y_t) \rightarrow (\bar{I}_t, \bar{V}_t, \bar{y}_t),
\label{eq:canonical}
\end{equation}
where $\bar{I}_t$, $\bar{V}_t$, and $\bar{y}_t$ are the canonicalized frame, event representation, and target, respectively.

\subsection{Multimodality End-to-End Frame-Event Hybrid Eye Tracking Transformer}
Frame and event inputs are embedded as
\begin{equation}
X_t^f = P_f(\bar{I}_t), \qquad X_t^e = P_e(\bar{V}_t),
\label{eq:embedding}
\end{equation}
and fused through the shared encoder:
\begin{equation}
Z_t = \mathrm{Enc}\big([X_t^f; X_t^e] + E_{\mathrm{pos}} + E_{\mathrm{mod}}\big).
\label{eq:fusion}
\end{equation}

The Track head predicts a residual displacement:
\begin{equation}
\hat{c}_t = \hat{c}_{t-1} + h_{\mathrm{trk}}(Z_t).
\label{eq:track_update}
\end{equation}

\begin{figure*}[!t]
\centering
\includegraphics[width=0.95\textwidth]{fig2_transformer.pdf}
\caption{Architecture of the proposed hybrid frame-event eye-tracking transformer. Frame and event inputs are tokenized, fused through a shared transformer encoder, and processed by decoupled Search, Track, and Mask heads.}
\label{fig:transformer}
\end{figure*}

\subsection{Two-Stage Search/Track Scheduler}
The scheduler switches modes according to confidence:
\begin{equation}
\text{Search} \rightarrow \text{Track} \quad \text{if } s_t > \tau_s,
\label{eq:s2t}
\end{equation}
\begin{equation}
\text{Track} \rightarrow \text{Search} \quad \text{if } q_t \le \tau_t.
\label{eq:t2s}
\end{equation}

\begin{figure}[!t]
\centering
\includegraphics[width=\columnwidth]{fig3_scheduler.pdf}
\caption{State transition diagram of the two-stage Search/Track scheduler. The host-side controller switches between global relocalization and low-latency local tracking based on confidence and geometric validity.}
\label{fig:scheduler}
\end{figure}

\subsection{Training Pipeline and Loss Function}
The overall loss is defined as
\begin{equation}
\mathcal{L} = \lambda_s \mathcal{L}_{\mathrm{search}}
+ \lambda_t \mathcal{L}_{\mathrm{track}}
+ \lambda_m \mathcal{L}_{\mathrm{mask}}
+ \lambda_g \mathcal{L}_{\mathrm{geo}}.
\label{eq:total_loss}
\end{equation}

\begin{figure}[!t]
\centering
\includegraphics[width=\columnwidth]{fig4_training.pdf}
\caption{Training pipeline of HBTXR. Dense annotations are used to canonicalize heterogeneous inputs and to supervise search, tracking, masking, and geometry-consistency objectives in an end-to-end manner.}
\label{fig:training}
\end{figure}

\section{The Proposed HBTXR Accelerator}
\subsection{System-Level CPU-FPGA Architecture}
The CPU handles sensing and runtime control, while the FPGA executes tokenization, transformer inference, and mode-aware refinement.

\subsection{Input Staging and Tokenization Engine}
Frame patches and event representations are staged in on-chip buffers and converted into patch tokens.

\subsection{Shared Transformer Engine}
The shared transformer engine supports multimodal token processing through pipelined attention and feed-forward blocks.

\subsection{Search/Track/Mask Head Engines}
Mode-specific heads consume the shared latent feature sequence without duplicating backbone computation.

\subsection{FSM-Based Runtime Scheduler}
An FPGA-side FSM controls mode-aware execution with low host intervention.

\subsection{CPU-FPGA Workload Partition and Dataflow}
The end-to-end latency is modeled as
\begin{equation}
T_{\mathrm{e2e}} = T_{\mathrm{host}}^{\mathrm{prep}} + T_{\mathrm{transfer}} + T_{\mathrm{fpga}}^{\mathrm{infer}} + T_{\mathrm{host}}^{\mathrm{post}}.
\label{eq:e2e_latency}
\end{equation}

\begin{figure*}[!t]
\centering
\includegraphics[width=0.95\textwidth]{fig5_architecture.pdf}
\caption{System-level CPU-FPGA architecture of HBTXR. The CPU handles sensing and runtime control, while the FPGA executes tokenization, transformer inference, head computation, and mode-aware refinement.}
\label{fig:hw_overview}
\end{figure*}

\begin{figure}[!t]
\centering
\includegraphics[width=\columnwidth]{fig6_engine.pdf}
\caption{Microarchitecture of the shared transformer engine. The engine supports multimodal token processing through pipelined projection, attention, aggregation, and feed-forward stages.}
\label{fig:engine}
\end{figure}

\begin{figure}[!t]
\centering
\includegraphics[width=\columnwidth]{fig7_reuse.pdf}
\caption{Streaming-based feature reuse and on-chip dataflow in Track mode. Cached frame-side features are reused while only event-dependent updates are recomputed to reduce latency and memory traffic.}
\label{fig:reuse}
\end{figure}

\begin{figure}[!t]
\centering
\includegraphics[width=\columnwidth]{fig8_fsm.pdf}
\caption{FSM-based runtime scheduling on FPGA. The accelerator-side FSM orchestrates Search and Track execution paths, buffer reuse, and head invocation with minimal host intervention.}
\label{fig:fsm}
\end{figure}

\begin{table}[!t]
\caption{Implementation Summary of the Proposed HBTXR System}
\label{tab:impl}
\centering
\begin{tabular}{ll}
\toprule
Item & Value \\
\midrule
Host CPU & [To be filled] \\
FPGA Device & [To be filled] \\
Clock Frequency & [To be filled] \\
Numeric Precision & [To be filled] \\
DMA Interface & [To be filled] \\
On-Chip Buffer Size & [To be filled] \\
Backbone & Partial DeiT-Tiny \\
Operating Modes & Search / Track \\
\bottomrule
\end{tabular}
\end{table}

\section{Experimental Results}
\subsection{Experimental Setup}
\subsubsection{Datasets and Splits}
Describe dataset composition, subject-disjoint protocol, and canonicalization.

\subsubsection{Training Settings}
Report optimizer, batch size, learning rate, augmentation, and loss weights.

\subsubsection{Hardware Platform}
Describe CPU-FPGA prototype and runtime settings.

\subsubsection{Metrics and Baselines}
Define pupil-center error, gaze angular error, latency, and comparison baselines.

\begin{equation}
\mathrm{Err}_{\mathrm{pupil}} = \frac{1}{N}\sum_{t=1}^{N} \|\hat{c}_t - c_t\|_2,
\label{eq:pupil_error}
\end{equation}

\begin{equation}
\mathrm{Err}_{\mathrm{gaze}} = \frac{1}{N}\sum_{t=1}^{N} \arccos \left( \frac{\hat{g}_t^\top g_t}{\|\hat{g}_t\|_2 \|g_t\|_2} \right).
\label{eq:gaze_error}
\end{equation}

\begin{table}[!t]
\caption{Datasets, Splits, and Evaluation Protocol}
\label{tab:dataset}
\centering
\begin{tabular}{lllllll}
\toprule
Dataset & Split & Subjects & Modality & Res. & Annotation & Usage \\
\midrule
[Dataset] & Train & [ ] & F+E & [ ] & Dense & Training \\
[Dataset] & Val & [ ] & F+E & [ ] & Dense & Validation \\
[Dataset] & Test & [ ] & F+E & [ ] & Dense & Testing \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[!t]
\caption{Baseline Methods Used in Experimental Comparison}
\label{tab:baselines}
\centering
\begin{tabular}{llllll}
\toprule
Method & Category & Modality & Task & Platform & Source \\
\midrule
Etracker & Frame & Frame & Gaze & CPU/GPU & [ ] \\
FACET & Event & Event & Pupil & CPU/GPU & [ ] \\
EX-Gaze & Hybrid & F+E & Gaze & Embedded & [ ] \\
EyeCoD & Deploy & Frame & Gaze & FPGA/ASIC & [ ] \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Evaluation of the Proposed Algorithm and Accelerator}
Present main Search/Track accuracy, latency, and runtime statistics.

\begin{table}[!t]
\caption{Main Performance of HBTXR Under Search and Track Modes}
\label{tab:main_results}
\centering
\begin{tabular}{lcccccc}
\toprule
Mode & Pupil (px) & Gaze (deg) & Search SR (\%) & Latency (ms) & Rate (Hz) & Notes \\
\midrule
Search & [ ] & [ ] & [ ] & [ ] & [ ] & Recovery \\
Track & 0.42 & 0.68 & -- & 0.57 & [ ] & Steady-state \\
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[!t]
\centering
\includegraphics[width=\columnwidth]{fig9_qualitative.pdf}
\caption{Qualitative results of HBTXR. Example outputs in Search and Track modes under different gaze directions, motion patterns, and recovery conditions.}
\label{fig:qualitative}
\end{figure}

\begin{figure}[!t]
\centering
\includegraphics[width=\columnwidth]{fig10_latency.pdf}
\caption{Latency breakdown of HBTXR. End-to-end runtime is decomposed into host preprocessing, transfer, FPGA inference, and host postprocessing for Search and Track modes.}
\label{fig:latency}
\end{figure}

\subsection{Comparison with Other Eye-Tracking Algorithms}
Compare HBTXR with frame-based, event-based, and hybrid baselines.

\begin{table*}[!t]
\caption{Comparison With Prior Eye-Tracking Algorithms}
\label{tab:algo_compare}
\centering
\begin{tabular}{lcccccccc}
\toprule
Method & Category & Modality & Pupil Err. & Gaze Err. & Latency & Reloc. & XR/On-device & E2E \\
\midrule
Etracker & Frame & Frame & [ ] & [ ] & [ ] & Yes & Partial & No \\
FACET & Event & Event & [ ] & [ ] & [ ] & No & Yes & Yes \\
EX-Gaze & Hybrid & F+E & [ ] & [ ] & [ ] & Partial & Yes & Partial \\
HBTXR & Hybrid & F+E & 0.42 & 0.68 & 0.57 ms & Yes & Yes & Yes \\
\bottomrule
\end{tabular}
\end{table*}

\begin{figure}[!t]
\centering
\includegraphics[width=\columnwidth]{fig11_tradeoff.pdf}
\caption{Accuracy-latency comparison with prior eye-tracking algorithms. HBTXR achieves a favorable trade-off between pupil/gaze accuracy and low-latency operation.}
\label{fig:tradeoff}
\end{figure}

\subsection{Comparison with Other Accelerators}
Compare HBTXR with prior FPGA/ASIC or deployment-oriented systems.

\begin{table*}[!t]
\caption{Comparison With Prior Eye-Tracking Accelerators and Deployment-Oriented Systems}
\label{tab:acc_compare}
\centering
\begin{tabular}{lccccccccc}
\toprule
Work & Platform & Modality & Backbone & Precision & Latency & Rate & Power/Energy & Scheduler & Notes \\
\midrule
EyeCoD & [ ] & Frame & CNN & [ ] & [ ] & [ ] & [ ] & No & Lensless \\
SEE & [ ] & Event & Sparse CNN & [ ] & [ ] & [ ] & [ ] & Partial & FPGA SoC \\
JaneEye & [ ] & Event & Custom & [ ] & [ ] & [ ] & [ ] & No & ASIC \\
HBTXR & FPGA & F+E & Transformer & [ ] & 0.57 ms & [ ] & [ ] & Yes & Hybrid \\
\bottomrule
\end{tabular}
\end{table*}

\begin{figure}[!t]
\centering
\includegraphics[width=\columnwidth]{fig12_accel_compare.pdf}
\caption{Comparison with prior accelerator and deployment-oriented systems. HBTXR provides competitive latency while supporting hybrid sensing, transformer inference, and runtime-adaptive execution.}
\label{fig:accel_compare}
\end{figure}

\subsection{Ablation Study}
Evaluate the effects of canonical supervision, decoupled heads, Search/Track scheduling, and feature reuse.

\begin{table}[!t]
\caption{Ablation Study of HBTXR}
\label{tab:ablation}
\centering
\begin{tabular}{lcccccccc}
\toprule
Variant & Canon. & Hybrid & Heads & S/T & Reuse & Pupil & Gaze & Lat. \\
\midrule
Full HBTXR & Yes & Yes & Yes & Yes & Yes & 0.42 & 0.68 & 0.57 \\
w/o Canon. & No & Yes & Yes & Yes & Yes & [ ] & [ ] & [ ] \\
w/o S/T & Yes & Yes & No & No & Yes & [ ] & [ ] & [ ] \\
w/o Reuse & Yes & Yes & Yes & Yes & No & [ ] & [ ] & [ ] \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Discussion}
Discuss strengths, failure modes, and deployment implications.

\section{Conclusion}
This paper presented HBTXR, an algorithm-hardware co-designed hybrid eye-tracking system for on-device XR applications. HBTXR jointly integrates dense-annotation-aware canonical supervision, an end-to-end hybrid transformer with decoupled heads, a two-stage Search/Track scheduler, and an FPGA-oriented accelerator architecture. Experimental results demonstrated accurate and low-latency operation, confirming the practicality of the proposed design for next-generation XR systems.

\section*{Acknowledgment}
This work was supported by [Funding Agency / Project Number].

\bibliographystyle{IEEEtran}
\bibliography{refs}

\end{document}
```

다음 작업으로 가장 자연스러운 것은 둘 중 하나다.

1. 위 skeleton에 맞춰 **Table I~VIII에 들어갈 실제 값/문구 템플릿**까지 채우기  
2. **Algorithm 1(Search/Track Scheduler)**와 **Fig. 1~12용 본문 참조 문장**까지 작성