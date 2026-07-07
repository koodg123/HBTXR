
\section{The Proposed HBTXR Accelerator}

\subsection{Design Challenges and Co-Design Principles}

The deployment objective of HBTXR is not merely to accelerate a generic transformer, but to support a hybrid eye-tracking workload whose operating modes exhibit fundamentally different execution characteristics. As discussed in Section~III, the Search branch performs robust relocalization and anchor refresh from frame-guided features, whereas the Track branch performs event-driven residual update conditioned on the previous state. This asymmetry leads to three hardware design challenges.

First, Search and Track have different dataflow requirements. Search is refresh-oriented and requires broader feature regeneration, whereas Track is reuse-oriented and benefits from persistent state and incremental update. A uniform inference path therefore leads either to unnecessary recomputation during Track mode or to insufficient global context during Search mode. Second, the shared transformer backbone introduces synchronization-heavy stages, especially in token interaction and branch fusion, while the decoupled heads and recurrent state update remain more amenable to lightweight streaming execution. Third, the target deployment setting imposes tight latency and memory constraints, making it undesirable to duplicate complete frame and event pipelines or to rely on repeated full-feature buffering.

These observations motivate an algorithm-hardware co-design strategy. HBTXR is structured so that the algorithmic decomposition in Section~III directly informs the execution structure of the accelerator. In particular, the shared-backbone and decoupled-head design enables the hardware to reuse a common transformer-oriented compute substrate, while the explicit Search/Track scheduler exposes two runtime modes that can be mapped differently in terms of buffering, feature reuse, and execution granularity. In this sense, the accelerator is not an after-the-fact implementation target, but an execution architecture derived from the semantics of the tracking algorithm.

\subsection{CPU-FPGA Workload Partition}

Fig.~\ref{fig:accel_overview} illustrates the overall deployment architecture of HBTXR. The system adopts a heterogeneous CPU-FPGA organization in which the host processor is responsible for control-centric functions and the FPGA is responsible for latency-dominant neural inference. This partition is chosen because Search/Track scheduling, state management, and control-policy updates are irregular and mode-dependent, whereas token generation, transformer inference, and head execution are regular and amenable to spatial acceleration.

More specifically, the CPU handles frame/event acquisition, runtime mode management, threshold comparison, closed-eye and validity handling, and final state dispatch to the application layer. The FPGA handles frame/event tokenization, modality adaptation, shared-backbone inference, Search/Event/Track/Mask head execution, and mode-specific feature reuse. Under this organization, Search and Track share the same accelerator substrate but differ in how aggressively previously computed features are retained and how much global context is refreshed.

This partition is consistent with the intended operation of HBTXR. Search mode requires stronger global robustness and therefore benefits from explicit host-visible control and a refresh-oriented execution path. Track mode, in contrast, is dominated by short-horizon low-latency updates and is therefore mapped to a state-persistent inference path with minimal control overhead. By separating host-side control from accelerator-side dense computation, HBTXR avoids conflating runtime policy with tensor execution while still preserving an efficient end-to-end deployment path.

\subsection{Mode-Aware Mixed-Granularity Execution Architecture}

A key design choice of the HBTXR accelerator is to avoid a single uniform execution granularity for the entire hybrid tracking pipeline. Instead, the proposed accelerator adopts a \emph{mode-aware mixed-granularity execution architecture}, shown in Fig.~\ref{fig:mixed_grain}. The main idea is that different stages of HBTXR exhibit different locality and synchronization characteristics and should therefore be mapped using different execution styles.

Stages that require broader synchronization across tokens or branches are executed using a \emph{phase-level execution path}. This path is appropriate for operations such as shared transformer processing, branch coordination, and full Search refresh, where the amount of reusable context is limited and inter-token dependency is relatively high. In contrast, stages that benefit from sequential locality and partial reuse are executed using a \emph{tile-stream execution path}. This path is suitable for Track-mode updates, lightweight head execution, and recurrent state propagation, where the dominant objective is to minimize idle gaps and buffering overhead while maintaining low-latency response.

This mixed-granularity organization serves two goals simultaneously. First, it reduces unnecessary storage associated with full refresh behavior in steady-state tracking. Second, it prevents a purely streaming design from being stalled by globally synchronized stages that require stronger coordination. In other words, the proposed accelerator does not attempt to force Search and Track into a single homogeneous pipeline. Instead, it uses the Search/Track distinction to define when global refresh is necessary and when incremental token-state processing is sufficient.

\subsection{Shared Backbone Engine and Mode-Specific Head Mapping}

The computational core of the accelerator is a shared transformer-oriented backbone engine that executes the common representation trunk of HBTXR. This design directly follows the shared-backbone model described in Section~III-C. Rather than implementing separate hardware paths for frame Search and event Track, the accelerator maintains a common backbone execution substrate and attaches mode-specific heads on top of it.

Let \(T_f\) and \(T_e\) denote the frame and event token sequences. The shared representation engine computes
\begin{equation}
\mathbf{H}_f = \mathcal{B}(T_f), \qquad
\mathbf{H}_e = \mathcal{B}(T_e),
\label{eq:shared_backbone_accel}
\end{equation}
where \(\mathcal{B}(\cdot)\) is realized by a common transformer engine. In Search mode, the accelerator emphasizes frame-guided refresh and generates search-oriented outputs from the shared representation. In Track mode, the accelerator prioritizes event-conditioned update and combines event features with the encoded previous state. This mode-specific mapping avoids duplicating the dominant compute blocks while preserving the functional separation of Search and Track.

The decoupled head structure is particularly useful from a hardware perspective. The Search head and Mask head are activated primarily during refresh-oriented operation, while the Track head dominates steady-state low-latency execution. The Event head provides an auxiliary event-side estimate that can be used for consistency and runtime monitoring without forcing the full Search path to run at every step. Consequently, the shared-backbone design improves both hardware reuse and scheduler visibility: the same representation engine can support different operating modes, and the runtime controller can invoke only the necessary head path based on the current state.

\subsection{Memory Organization and Reuse-Oriented Dataflow}

Because the target application is on-device XR eye tracking, memory movement is often more critical than arithmetic count. The proposed accelerator therefore uses a memory organization that distinguishes between \emph{refresh-oriented buffering} and \emph{stream-oriented reuse}. Search mode uses a refresh buffer pair to support broader feature regeneration and robust relocalization. Track mode uses a streaming token queue and persistent state buffers so that recently computed context can be reused without re-materializing the entire inference path.

This organization is shown conceptually in Fig.~\ref{fig:search_track_timeline}. During Search mode, frame and event inputs are staged into a refresh path that regenerates the required shared features. During Track mode, the accelerator reuses persistent features and updates only the event-conditioned path together with the previous-state encoding. In this way, the dominant cost of Search is paid only when anchor refresh is required, while Track remains lightweight and reuse-oriented.

The resulting dataflow is mode-aware rather than operator-centric. Search follows a refresh-heavy path with stronger global synchronization, whereas Track follows a state-persistent update path with lower buffering pressure. This difference is important because it allows HBTXR to reduce inter-stage idle gaps during low-latency operation without sacrificing the robustness of periodic relocalization. The objective is therefore not simply to maximize raw throughput, but to minimize the end-to-end latency of the tracking loop under the actual Search/Track runtime pattern.

\subsection{Runtime Control and Scheduling Support}

To support mode-aware execution, the accelerator is coupled with an FSM-based runtime controller that interprets the host-side Search/Track decisions and dispatches the corresponding execution path. The controller is lightweight but functionally important: it determines whether the current step should invoke refresh-oriented Search, reuse-oriented Track, or auxiliary branch evaluation for confidence assessment. This design matches the scheduler semantics introduced in Section~III-E.

From the hardware viewpoint, the controller manages three classes of signals: 1) mode-control signals from the host, 2) data-valid and buffer-status signals within the accelerator, and 3) branch-confidence and output-ready signals used to synchronize execution and state update. By keeping this controller explicit, HBTXR avoids implicit control hidden inside a fixed inference path. Instead, runtime mode changes become first-class events in the execution architecture.

This control structure is one of the main distinctions between HBTXR and a conventional transformer accelerator. In a generic accelerator, the dominant concern is often maximizing uniform utilization over a fixed computation graph. In HBTXR, however, the execution graph itself is mode-dependent. The controller therefore does more than sequence operators; it realizes the runtime semantics of Search and Track as two distinct operating states of the system.

\subsection{Latency Model and Deployment Scope}

The end-to-end latency of HBTXR can be decomposed as
\begin{equation}
T_{\mathrm{e2e}}
=
T_{\mathrm{host}}
+
T_{\mathrm{transfer}}
+
T_{\mathrm{exec}}
+
T_{\mathrm{sync}},
\label{eq:latency_decomp}
\end{equation}
where \(T_{\mathrm{host}}\) denotes host-side control and scheduling time, \(T_{\mathrm{transfer}}\) denotes host-device communication, \(T_{\mathrm{exec}}\) denotes accelerator execution time, and \(T_{\mathrm{sync}}\) denotes runtime synchronization overhead. Because the Search and Track paths have different execution structures, their latencies are also different. Let \(T_{\mathrm{search}}\) and \(T_{\mathrm{track}}\) denote the two mode-specific latencies. The average runtime under scheduler-driven operation can be expressed as
\begin{equation}
T_{\mathrm{avg}}
=
\alpha T_{\mathrm{search}}
+
(1-\alpha)T_{\mathrm{track}},
\label{eq:avg_latency}
\end{equation}
where \(\alpha\) is the fraction of Search invocations. This formulation highlights why mode-aware hardware design matters: reducing \(T_{\mathrm{track}}\) alone is insufficient if Search remains too expensive, and reducing Search cost alone is wasteful if Track cannot exploit reuse effectively.

Accordingly, the purpose of the proposed accelerator is not to maximize a synthetic throughput benchmark in isolation, but to provide a deployment-oriented execution architecture that aligns with the actual Search/Track operating behavior of HBTXR. The architecture therefore emphasizes three principles: 1) shared backbone reuse, 2) asymmetric handling of Search and Track, and 3) minimization of memory and synchronization overhead through mixed-granularity execution.

Finally, the scope of the present hardware contribution should be understood as a co-design architecture and deployment mapping for HBTXR. The proposed accelerator section establishes how the hybrid tracking algorithm can be organized into a practical host-device execution path and how Search and Track can be mapped differently without fragmenting the model structure. The quantitative implications of this design are evaluated in Section~V from the perspectives of latency, accuracy, and deployment-oriented efficiency.
