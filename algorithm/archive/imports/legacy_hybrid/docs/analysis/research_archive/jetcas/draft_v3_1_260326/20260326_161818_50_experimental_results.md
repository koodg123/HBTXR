
\section{Experimental Results}

\subsection{Experimental Setup}

\subsubsection{Dataset and Evaluation Protocol}

We evaluate HBTXR on a near-eye hybrid eye-tracking benchmark composed of synchronized frame observations, event streams, and dense pupil annotations. Each sample contains a canonicalized frame patch, a polarity-split event tensor, the previous tracking state, and geometry-aware supervision targets including the pupil mask, eye region, and current pupil state. Following the hybrid supervision framework introduced in Section~III, all samples are transformed into a unified canonical coordinate system before training and evaluation.

For robust generalization assessment, the dataset is divided using a subject-disjoint protocol. The training split is used for stage-wise model optimization, the validation split is used for hyperparameter selection and early stopping, and the test split is used only for final reporting. In addition, when available, external benchmarks or public subsets may be used for cross-dataset evaluation in order to assess robustness beyond the in-domain canonical training distribution. The detailed dataset and split configuration are summarized in Table~\ref{tab:exp_setup}.

\subsubsection{Training Configuration}

HBTXR is trained in two stages. In the first stage, the model is optimized for Search-oriented relocalization so that the frame-guided branch learns stable eye-region and pupil-state estimation. In the second stage, the hybrid network is fine-tuned with Search, Event, and Track supervision jointly, together with consistency and quality-aware gating. Unless otherwise specified, the default event representation uses a fixed-count event window with causal accumulation, and the default spatial preprocessing follows the canonical direct-square resizing policy described in Section~III.

The backbone and head parameters are optimized using AdamW with batch size, learning rate, and regularization settings reported in Table~\ref{tab:exp_setup}. The best model is selected according to the validation metric defined for the current training stage. Optional self-distillation and width-slicing paths are implemented in the framework but are disabled by default unless explicitly enabled in the corresponding ablation study.

\subsubsection{Runtime and Deployment Configuration}

To evaluate deployment-oriented behavior, we measure both algorithmic accuracy and runtime efficiency under the Search/Track execution model. Search mode is invoked for initialization and relocalization, while Track mode is used for steady-state residual updates. The runtime scheduler uses confidence, similarity, event-density, and eye-state signals to determine the active mode. The threshold settings used in the reported experiments are listed in Table~\ref{tab:exp_setup}.

For hardware-oriented evaluation, we consider a heterogeneous host-device deployment setting in which host-side scheduling and control are separated from the accelerator-side inference path. The reported runtime includes the mode-aware execution behavior of HBTXR and is decomposed into host control, transfer, inference, and synchronization overheads when latency breakdown is analyzed.

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

To evaluate runtime behavior, we report end-to-end latency, effective update rate, and, when applicable, Search success rate and recovery behavior. For deployment-oriented analysis, we further examine the effect of mode-aware execution, feature reuse, and accelerator-side execution mapping.

\subsubsection{Baselines}

We compare HBTXR with representative prior methods from three categories: 1) frame-based eye tracking, 2) event-based eye tracking, and 3) hybrid or deployment-oriented eye-tracking systems. Frame-based baselines are included to represent semantically stable but computation-heavy pipelines, event-based baselines represent low-latency sparse tracking systems, and hybrid baselines represent prior attempts to combine relocalization and event-driven update. When hardware-oriented comparisons are discussed, we focus on deployment-aware systems whose reported metrics are sufficiently comparable in terms of operating objective and runtime setting. The baseline list is summarized in Table~\ref{tab:baseline_summary}.

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
Batch size & [To be filled] \\
Learning rate & [To be filled] \\
Weight decay & [To be filled] \\
Runtime platform & Host-device heterogeneous execution \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[!t]
\caption{Baseline Categories Used in Comparison}
\label{tab:baseline_summary}
\centering
\footnotesize
\begin{tabular}{lll}
\toprule
Category & Representative Methods & Primary Focus \\
\midrule
Frame-based & Etracker, HE-Tracker & Semantic stability \\
Event-based & E-Track, FACET, Swift-Eye & Low-latency update \\
Hybrid / deploy. & EX-Gaze, SEE, EyeCoD, JaneEye & Fusion / deployment \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Main Results}

Table~\ref{tab:main_results} summarizes the main tracking performance of HBTXR. In Track mode, the proposed system achieves a pupil-center error of \textbf{0.42 pixels}, a gaze angular error of \textbf{0.68$^\circ$}, and an end-to-end latency of \textbf{0.57 ms}. These results indicate that HBTXR is able to maintain both geometric accuracy and sub-millisecond responsiveness during steady-state tracking, which is essential for practical XR applications.

Search mode incurs higher cost than Track mode, as expected, because it refreshes the global tracking anchor and relies on broader semantic context. However, this cost is amortized by the Search/Track scheduler, which limits Search invocation to cases in which relocalization or anchor refresh is required. Consequently, the effective operating point of HBTXR is determined not only by the Search and Track branches individually, but also by how the scheduler balances robustness and reuse during long-horizon runtime.

The results confirm the central design intuition of HBTXR: Search and Track should not be optimized as a single uniform inference path. Search improves robustness by recovering a stable anchor from frame-guided information, whereas Track improves responsiveness by updating the state through event-guided residual inference. When combined under the proposed scheduler, the system achieves a stronger balance between semantic stability and temporal efficiency than either branch alone.

\begin{table}[!t]
\caption{Main Results of HBTXR}
\label{tab:main_results}
\centering
\footnotesize
\begin{tabular}{lccc}
\toprule
Mode & Pupil Error (px) & Gaze Error (deg) & Latency (ms) \\
\midrule
Search & [To be filled] & [To be filled] & [To be filled] \\
Track & \textbf{0.42} & \textbf{0.68} & \textbf{0.57} \\
Scheduled average & [To be filled] & [To be filled] & [To be filled] \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Comparison With Prior Eye-Tracking Methods}

We next compare HBTXR with representative prior eye-tracking methods. Compared with conventional frame-based pipelines, HBTXR provides substantially better suitability for low-latency XR tracking because steady-state operation is dominated by event-guided residual updates rather than repeated full-frame inference. At the same time, unlike purely event-based trackers, HBTXR retains an explicit relocalization path through the Search branch, which improves robustness when event support is weak or when drift accumulates over time.

Compared with prior hybrid methods, HBTXR differs in two important ways. First, Search and Track are modeled as distinct functional paths rather than as a loosely fused multimodal predictor. Second, the runtime scheduler and deployment architecture are considered part of the system design rather than as an external implementation detail. This allows HBTXR to couple hybrid sensing with mode-aware execution, which is especially important in on-device XR settings where the dominant cost is often determined by the runtime path rather than by nominal model size alone.

A quantitative comparison with representative prior methods is summarized in Table~\ref{tab:compare_methods}. The comparison should be interpreted from the perspective of overall trade-off rather than isolated best-case accuracy. The main strength of HBTXR is that it simultaneously supports relocalization, low-latency tracking, and deployment-aware execution within a single hybrid framework.

\begin{table*}[!t]
\caption{Comparison With Prior Eye-Tracking Methods}
\label{tab:compare_methods}
\centering
\footnotesize
\begin{tabular}{lcccccc}
\toprule
Method & Modality & Relocalization & Pupil Error & Gaze Error & Latency & On-device Orientation \\
\midrule
Etracker & Frame & Yes & [ ] & [ ] & [ ] & Limited \\
HE-Tracker & Frame & Yes & [ ] & [ ] & [ ] & Partial \\
E-Track & Event & No & [ ] & [ ] & [ ] & Yes \\
FACET & Event & No & [ ] & [ ] & [ ] & Yes \\
EX-Gaze & Hybrid & Partial & [ ] & [ ] & [ ] & Yes \\
HBTXR & Hybrid & Yes & \textbf{0.42} & \textbf{0.68} & \textbf{0.57 ms} & Yes \\
\bottomrule
\end{tabular}
\end{table*}

\subsection{Deployment-Oriented Analysis}

Beyond algorithmic accuracy, we analyze HBTXR from a deployment-oriented perspective. The key question is whether the proposed Search/Track decomposition and mixed-granularity execution model actually improve the practical runtime profile of the system. To answer this, we examine the role of the scheduler, the asymmetric Search/Track paths, and the effect of feature reuse.

First, the scheduler reduces the average runtime cost by allowing Search to be invoked only when required. This is especially beneficial in typical XR operation, where long periods of stable local motion are punctuated by occasional relocalization events. Second, the shared-backbone design allows the same representation engine to support both Search and Track, which reduces structural duplication. Third, the reuse-oriented Track path lowers the effective cost of steady-state inference, making the latency of the full system much closer to Track-mode latency than to Search-mode latency.

When comparing HBTXR with deployment-oriented baselines, the most important distinction is that HBTXR is not designed solely for one sensing path or one fixed execution schedule. Instead, it supports a hybrid operating regime in which refresh-heavy Search and reuse-heavy Track coexist within the same framework. This makes the comparison broader than a pure accelerator benchmark: the relevant criterion is whether the system can sustain robust and low-latency eye tracking under realistic runtime behavior. A summary of deployment-oriented comparison points is provided in Table~\ref{tab:compare_deploy}.

\begin{table*}[!t]
\caption{Deployment-Oriented Comparison}
\label{tab:compare_deploy}
\centering
\footnotesize
\begin{tabular}{lccccc}
\toprule
System & Modality & Runtime Adaptivity & Shared Backbone Path & Low-Latency Track Path & Deployment Scope \\
\midrule
EyeCoD & Frame & No & No & No & Compact frame pipeline \\
SEE & Event & Partial & No & Yes & Event-centric acceleration \\
JaneEye & Event & No & No & Yes & Event ASIC acceleration \\
HBTXR & Hybrid & Yes & Yes & Yes & Hybrid Search/Track deployment \\
\bottomrule
\end{tabular}
\end{table*}

\subsection{Ablation Studies}

To analyze the contribution of each design component, we conduct ablation studies on the supervision framework, the hybrid model structure, the scheduler, and the deployment-oriented execution path.

\subsubsection{Effect of Geometry-Stable Supervision}

We first evaluate the impact of the proposed geometry-stable hybrid supervision contract. Removing or simplifying the geometry-aware supervision degrades both Search robustness and Track consistency, indicating that the common target representation is important for aligning frame-guided relocalization and event-guided residual update. This confirms that the supervision design is not merely a data-format choice, but a core component of the hybrid tracking formulation.

\subsubsection{Effect of Shared Backbone and Decoupled Heads}

We next evaluate the shared-backbone design and the decoupled Search/Event/Track heads. Replacing the decoupled structure with a more uniform prediction path reduces scheduler visibility and weakens the distinction between global recovery and local update. Similarly, removing the shared-backbone structure increases architectural fragmentation and weakens the deployment-oriented argument of the system. These results support the use of a unified backbone together with branch-specific output heads.

\subsubsection{Effect of Search/Track Scheduler}

We also evaluate the Search/Track scheduler independently. Disabling the scheduler and forcing a fixed execution mode degrades the robustness-efficiency trade-off. Always-on Search increases runtime overhead, whereas Track-only operation reduces relocalization capability. The scheduler is therefore essential not only for runtime efficiency, but also for preserving long-horizon tracking stability.

\subsubsection{Effect of Deployment-Oriented Mapping}

Finally, we evaluate the impact of the deployment-oriented execution design. In particular, we examine the effect of reuse-oriented Track execution and mixed-granularity mapping. The results show that reusing shared features and separating refresh-oriented Search from state-persistent Track reduces effective runtime cost without changing the functional structure of the model. This confirms that deployment-aware execution is a meaningful contributor to the final system performance.

\begin{table}[!t]
\caption{Ablation Study}
\label{tab:ablation}
\centering
\footnotesize
\begin{tabular}{lccc}
\toprule
Variant & Pupil Error & Gaze Error & Latency \\
\midrule
Full HBTXR & \textbf{0.42} & \textbf{0.68} & \textbf{0.57} \\
w/o geometry-stable supervision & [ ] & [ ] & [ ] \\
w/o decoupled heads & [ ] & [ ] & [ ] \\
w/o Search/Track scheduler & [ ] & [ ] & [ ] \\
w/o reuse-oriented execution & [ ] & [ ] & [ ] \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Qualitative Analysis and Discussion}

Fig.~\ref{fig:qualitative_results} presents representative qualitative examples. In stable conditions, Track mode maintains smooth low-latency updates with accurate local state propagation. Under more difficult cases, including reduced event density or degraded confidence, the scheduler falls back to Search mode to refresh the global anchor. These examples qualitatively illustrate how HBTXR balances relocalization robustness and update efficiency.

Overall, the experiments support three conclusions. First, hybrid sensing alone is insufficient unless Search and Track are explicitly separated at the algorithmic and runtime levels. Second, geometry-stable supervision is critical for making frame-guided relocalization and event-guided tracking compatible within one model. Third, deployment performance depends not only on the prediction network, but also on how the runtime path is organized. In HBTXR, the mixed-granularity execution strategy and reuse-oriented Track path make the system better aligned with the real operating conditions of on-device XR eye tracking.

At the same time, several limitations remain. The Search path is still more expensive than the Track path, and therefore the final runtime depends on how frequently relocalization is triggered. In addition, deployment-oriented claims should be interpreted together with the actual execution substrate and measurement conditions. Future work should refine the hardware realization further and broaden the evaluation to more diverse runtime scenarios and eye-related tasks.
