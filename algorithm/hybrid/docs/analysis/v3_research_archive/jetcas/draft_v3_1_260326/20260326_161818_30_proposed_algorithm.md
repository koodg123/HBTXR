
\section{The Proposed HBTXR Algorithm}

\subsection{Problem Formulation}

As illustrated in Fig.~\ref{fig:overview}, HBTXR is designed for on-device XR eye tracking under a hybrid sensing setting composed of a frame stream, an event stream, and a recurrent tracking state. Let \(I_t \in \mathbb{R}^{H \times W}\) denote the frame acquired at time \(t\), and let
\begin{equation}
\mathcal{E}_t = \left\{ e_k = (x_k, y_k, p_k, \tau_k) \right\}_{k=1}^{N_t}
\label{eq:event_stream}
\end{equation}
denote the event set collected over the current temporal window, where \((x_k,y_k)\) is the event location, \(p_k \in \{-1,+1\}\) is the polarity, and \(\tau_k\) is the timestamp. The event stream is converted into a polarity-split event tensor \(V_t \in \mathbb{R}^{2 \times H \times W}\) through a causal accumulation operator
\begin{equation}
V_t = \Phi(\mathcal{E}_t; \omega_t),
\label{eq:event_tensor}
\end{equation}
where \(\omega_t\) denotes the event-window construction policy such as fixed-count or time-bin slicing.

The objective of HBTXR is to estimate the current pupil state
\begin{equation}
\mathbf{s}_t = [x_t, y_t, a_t, b_t, u_t, v_t] \in \mathbb{R}^{6},
\label{eq:state_def}
\end{equation}
where \((x_t,y_t)\) is the pupil center, \((a_t,b_t)\) are the ellipse axes, and \((u_t,v_t)\) is a trigonometric orientation encoding. In particular, HBTXR adopts the relation
\begin{equation}
[u_t, v_t] = [\sin(2\theta_t), \cos(2\theta_t)],
\label{eq:uv_def}
\end{equation}
which avoids discontinuity around the angular wrap-around boundary and enables stable regression of ellipse orientation.

A central design choice of HBTXR is to formulate hybrid eye tracking as a \emph{Search-and-Track} problem rather than as a single monolithic prediction task. The \emph{Search} path is responsible for robust relocalization and anchor refresh using frame-based semantic cues, while the \emph{Track} path performs low-latency residual update from the previous state using event observations. In addition, an event-conditioned estimation branch is introduced to provide an auxiliary event-side state hypothesis that supports cross-branch consistency and runtime confidence assessment. Under this formulation, HBTXR jointly predicts a global search state, an event-conditioned state, and a residual track state, and then selects the active update path through a runtime-aware scheduler.

\subsection{Geometry-Stable Hybrid Supervision}

A major challenge in hybrid frame-event tracking is that frame cues, event cues, and recurrent state updates must be trained against a common geometric target. To address this issue, HBTXR adopts a dense-annotation-aware canonical supervision framework. Each raw sample is converted into a canonical training representation containing an eye region, pupil mask, ellipse annotation, and quality metadata. Let \(\mathcal{T}(\cdot)\) denote the canonicalization operator. The training sample is constructed as
\begin{equation}
(\bar{I}_t, \bar{V}_t, \bar{\mathbf{s}}_{t-1}, \bar{\mathbf{s}}_t, \bar{M}_t, \bar{B}_t, \bar{\xi}_t)
=
\mathcal{T}(I_t, \mathcal{E}_t, \mathbf{s}_{t-1}, \mathbf{s}_t, M_t, B_t, \xi_t),
\label{eq:canonical_contract}
\end{equation}
where \(\bar{I}_t\) and \(\bar{V}_t\) denote the canonicalized frame and event inputs, \(\bar{M}_t\) is the canonical pupil mask, \(\bar{B}_t\) is the canonical eye-region target, and \(\bar{\xi}_t\) collects supervision-quality signals such as annotation confidence, mask validity, closed-eye flag, and track validity.

The resulting sample contract provides three important properties. First, Search and Track are trained against the same canonical state representation, which prevents the two branches from drifting toward incompatible geometries. Second, the eye region, mask, ellipse, and state targets are spatially aligned under a single transform, so the supervision is consistent across modalities. Third, quality and validity flags are propagated explicitly to the loss functions and the scheduler, allowing the model to distinguish between reliable tracking conditions and samples that should be partially masked during training.

This supervision contract is not merely a preprocessing convenience. It is the mechanism that allows frame-guided relocalization, event-guided residual tracking, and runtime switching logic to share a common geometric interpretation. Without such a contract, the Search and Track branches would have to be aligned only implicitly through optimization, which is particularly fragile in hybrid settings with heterogeneous sensing characteristics.

\subsection{Hybrid Search-Track Transformer}

The proposed HBTXR network is shown in Fig.~\ref{fig:hbtxr_algorithm}. The model uses modality-specific embedding layers and adapters, followed by a shared transformer backbone. Unlike architectures that fully fuse frame and event tensors at the input level, HBTXR preserves modality-specific front-ends and reuses a common token-processing trunk so that search-oriented frame inference and track-oriented event inference can be expressed under a unified backbone.

For the frame branch, the canonical frame \(\bar{I}_t\) is converted into patch tokens through a frame patch embedding module:
\begin{equation}
\mathbf{Z}^{f}_t = \mathcal{B}\!\left(\mathcal{A}_f\!\left(\mathcal{P}_f(\bar{I}_t)\right)\right),
\label{eq:frame_embed}
\end{equation}
where \(\mathcal{P}_f(\cdot)\) denotes frame patch embedding, \(\mathcal{A}_f(\cdot)\) is a modality adapter, and \(\mathcal{B}(\cdot)\) is the shared Partial DeiT-Tiny backbone~\cite{Touvron2021DeiT}. For the event branch, the canonical event tensor \(\bar{V}_t\) is processed similarly:
\begin{equation}
\mathbf{Z}^{e}_t = \mathcal{B}\!\left(\mathcal{A}_e\!\left(\mathcal{P}_e(\bar{V}_t)\right)\right).
\label{eq:event_embed}
\end{equation}
This design keeps the modality-dependent signal extraction lightweight while consolidating the dominant representation capacity into a shared transformer trunk.

On top of the shared backbone, HBTXR uses decoupled prediction heads:
\begin{itemize}
\item an \emph{Eye Region Head} for coarse eye-region regression,
\item a \emph{Pupil Search Head} for frame-guided global relocalization,
\item an \emph{Event Search Head} for event-conditioned state estimation,
\item a \emph{Pupil Track Head} for previous-state-conditioned residual prediction,
\item a \emph{Search Mask Head} for dense pupil-mask prediction, and
\item an \emph{Auxiliary State Head} for auxiliary supervision and runtime support.
\end{itemize}

Let \(\mathbf{p}^{f}_t\) and \(\mathbf{p}^{e}_t\) denote the pooled frame and event features, respectively. The search branch predicts
\begin{equation}
\hat{\mathbf{y}}^{s}_t = h_s(\mathbf{p}^{f}_t),
\qquad
\hat{\mathbf{s}}^{s}_t = \hat{\mathbf{y}}^{s}_t[1{:}6],
\label{eq:search_pred}
\end{equation}
while the event branch predicts
\begin{equation}
\hat{\mathbf{y}}^{e}_t = h_e(\mathbf{p}^{e}_t),
\qquad
\hat{\mathbf{s}}^{e}_t = \hat{\mathbf{y}}^{e}_t[1{:}6].
\label{eq:event_pred}
\end{equation}
The dense mask branch predicts
\begin{equation}
\hat{M}_t = h_m(\mathbf{Z}^{f}_t),
\label{eq:mask_pred}
\end{equation}
which provides both geometric regularization and a confidence-aware auxiliary signal for relocalization quality.

\subsection{Search Branch, Track Branch, and State Update}

The Search branch is designed for global relocalization. It is frame-guided because frame observations are more semantically stable under challenging event conditions. In contrast, the Track branch is designed for high-rate residual update. It combines the event feature \(\mathbf{p}^{e}_t\) with an encoded previous state \(\psi(\bar{\mathbf{s}}_{t-1})\) and predicts a residual vector:
\begin{equation}
\Delta \hat{\mathbf{r}}_t
=
h_t\!\left(
\left[
\mathbf{p}^{e}_t ; \psi(\bar{\mathbf{s}}_{t-1})
\right]
\right),
\label{eq:track_residual}
\end{equation}
where
\begin{equation}
\Delta \hat{\mathbf{r}}_t =
[\Delta \hat{x}_t, \Delta \hat{y}_t, \Delta \widehat{\log a}_t, \Delta \widehat{\log b}_t, \Delta \hat{u}_t, \Delta \hat{v}_t, \hat{c}^{trk}_t, \hat{q}^{trk}_t].
\label{eq:track_output}
\end{equation}
Here, \(\hat{c}^{trk}_t\) denotes track confidence and \(\hat{q}^{trk}_t\) denotes track quality.

The decoded tracking state is obtained as
\begin{equation}
\hat{x}_t = x_{t-1} + \Delta \hat{x}_t, \qquad
\hat{y}_t = y_{t-1} + \Delta \hat{y}_t,
\label{eq:track_xy}
\end{equation}
\begin{equation}
\hat{a}_t = a_{t-1}\exp(\Delta \widehat{\log a}_t), \qquad
\hat{b}_t = b_{t-1}\exp(\Delta \widehat{\log b}_t),
\label{eq:track_ab}
\end{equation}
\begin{equation}
[\hat{u}_t,\hat{v}_t]
=
\frac{
[u_{t-1} + \Delta \hat{u}_t,\; v_{t-1} + \Delta \hat{v}_t]
}{
\left\|
[u_{t-1} + \Delta \hat{u}_t,\; v_{t-1} + \Delta \hat{v}_t]
\right\|_2 + \epsilon
}.
\label{eq:track_uv}
\end{equation}

This residual parameterization is particularly suitable for short-horizon event-driven updates. Instead of forcing the event branch to re-estimate a full global state from scratch at every step, HBTXR models tracking as a constrained local update around the current anchor. This improves temporal stability and reduces the burden on the event path during steady-state operation.

\subsection{Runtime-Aware Search/Track Scheduler}

A key component of HBTXR is the explicit Search/Track scheduler. Rather than using all branches uniformly at every step, the scheduler decides whether the system should remain in low-latency Track mode or fall back to Search mode for relocalization. This decision is based on search confidence, track confidence, track quality, cross-branch similarity, event density, and eye-state validity.

Let \(c^s_t\) be the search confidence, \(c^{trk}_t\) the tracking confidence, \(q^{trk}_t\) the tracking quality, \(\rho_t\) the similarity between search and track states, \(d_t\) the event density, and \(o_t\) a binary closed-eye indicator. The scheduler transitions from Search to Track when the search estimate is reliable:
\begin{equation}
\text{Search} \rightarrow \text{Track}
\quad \text{if} \quad
c^s_t > \tau_s.
\label{eq:scheduler_st}
\end{equation}
Conversely, it re-enters Search mode if the tracking path becomes unreliable:
\begin{equation}
\text{Track} \rightarrow \text{Search}
\quad \text{if} \quad
\left(c^{trk}_t \le \tau_t\right)
\;\vee\;
\left(q^{trk}_t \le \tau_q\right)
\;\vee\;
\left(\rho_t \le \tau_{\rho}\right)
\;\vee\;
\left(d_t \le \tau_d\right)
\;\vee\;
(o_t = 1).
\label{eq:scheduler_ts}
\end{equation}

In practice, the scheduler serves two roles. Algorithmically, it enforces the distinction between global relocalization and local residual update. Operationally, it exposes a runtime structure that can be mapped naturally to a host-controlled accelerator path. This explicit scheduler is therefore not only a tracking policy, but also a key part of the algorithm-hardware co-design philosophy of HBTXR.

\subsection{Training Strategy and Loss Design}

HBTXR is trained in two stages. In the first stage, the model is trained primarily for Search-oriented relocalization so that the frame branch learns robust global geometric estimation. In the second stage, the hybrid network is fine-tuned with Track supervision, event-conditioned estimation, and cross-branch consistency. This stage-wise strategy is important because relocalization and residual tracking have different optimization characteristics and failure modes.

The total training objective is written as
\begin{equation}
\mathcal{L}
=
\lambda_{eye}\mathcal{L}_{eye}
+
\lambda_{mask}\mathcal{L}_{mask}
+
\lambda_{search}\mathcal{L}_{search}
+
\lambda_{event}\mathcal{L}_{event}
+
\lambda_{track}\mathcal{L}_{track}
+
\lambda_{cons}\mathcal{L}_{cons}
+
\lambda_{ctr}\mathcal{L}_{ctr}
+
\lambda_{aux}\mathcal{L}_{aux},
\label{eq:total_loss}
\end{equation}
where \(\mathcal{L}_{eye}\) is the eye-region loss, \(\mathcal{L}_{mask}\) is the pupil-mask loss, \(\mathcal{L}_{search}\) supervises frame-based search, \(\mathcal{L}_{event}\) supervises event-conditioned estimation, \(\mathcal{L}_{track}\) supervises residual tracking, \(\mathcal{L}_{cons}\) enforces cross-branch consistency, \(\mathcal{L}_{ctr}\) constrains the predicted center within the valid search region, and \(\mathcal{L}_{aux}\) denotes auxiliary supervision.

For both search and tracking, HBTXR uses geometry-aware supervision in addition to standard center and axis regression. Let \(\mu(\mathbf{s})\) and \(\Sigma(\mathbf{s})\) denote the ellipse center and covariance implied by state \(\mathbf{s}\). A geometry-aware term is defined as
\begin{equation}
\mathcal{L}_{geo}(\hat{\mathbf{s}}, \mathbf{s})
=
\left\|
\mu(\hat{\mathbf{s}})-\mu(\mathbf{s})
\right\|_2^2
+
\left\|
\Sigma(\hat{\mathbf{s}})^{1/2}-\Sigma(\mathbf{s})^{1/2}
\right\|_F^2,
\label{eq:geo_loss}
\end{equation}
which stabilizes the full ellipse geometry rather than supervising only the center location. In the actual implementation, this geometry-aware loss is combined with coordinate, axis, trigonometric, confidence, and quality terms.

Finally, HBTXR applies quality-aware loss gating based on annotation confidence, mask validity, tracking validity, and closed-eye status. Geometry-heavy terms are suppressed when the corresponding supervision is unreliable, whereas confidence-aware terms can remain active. This prevents noisy labels or invalid conditions from dominating optimization and further aligns the training objective with the intended runtime behavior.

Although the current framework also supports optional teacher-student self-distillation and width-slicing extensions, these mechanisms are treated as auxiliary enhancements rather than core components of the proposed algorithm. The essential algorithmic contribution of HBTXR remains the joint design of geometry-stable supervision, hybrid Search/Track modeling, and runtime-aware scheduling.
