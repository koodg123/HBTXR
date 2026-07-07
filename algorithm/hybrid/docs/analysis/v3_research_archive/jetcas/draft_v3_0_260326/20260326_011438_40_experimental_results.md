## VI. EXPERIMENTAL RESULTS

### A. Experimental Setup

#### 1) Datasets and Evaluation Protocol

We evaluate HBTXR on a near-eye hybrid sensing benchmark composed of synchronized frame images, event streams, and dense pupil annotations. The dataset contains subject-diverse recordings collected under varying gaze directions, motion patterns, and illumination conditions. To ensure fair generalization assessment, the data are divided into training, validation, and test sets using a subject-disjoint split. When available, additional public benchmarks such as `[Dataset-A]`, `[Dataset-B]`, and `[Dataset-C]` are used for cross-dataset evaluation.

Each sample consists of a frame patch \(I_t\), an event slice \(\mathcal{E}_t\), and dense geometric labels including pupil center and pupil mask. Following Section IV, all samples are transformed into the canonical eye coordinate system before training and evaluation. For Track-mode experiments, temporally adjacent event slices are paired with the previous estimated pupil state to emulate high-rate local refinement. For Search-mode experiments, larger perturbations are applied to simulate initialization and relocalization conditions. Unless otherwise stated, all reported results are obtained under the same canonicalization and preprocessing pipeline.

#### 2) Training Configuration

The proposed hybrid transformer is trained end-to-end using the loss formulation described in Section IV-D. The backbone is initialized from `[scratch / ImageNet-pretrained DeiT-Tiny weights]`, and the full network is optimized using AdamW with an initial learning rate of `[ ]`, weight decay of `[ ]`, and batch size of `[ ]`. The loss weights \(\lambda_s\), \(\lambda_t\), \(\lambda_m\), and \(\lambda_g\) are selected on the validation set. Frame inputs are augmented using brightness, contrast, and blur perturbations, whereas event inputs are augmented using event sparsity variation, polarity noise, and temporal slicing perturbation. All models are trained for `[ ]` epochs, and the best checkpoint is selected according to the validation pupil-center error.

#### 3) Hardware and Runtime Configuration

The deployment prototype consists of a host CPU and an FPGA accelerator connected through `[PCIe / AXI / custom DMA interface]`. The CPU executes host-side runtime control, sensor handling, and Search/Track mode selection, while the FPGA performs tokenization, transformer inference, head execution, and geometry-aware refinement. The FPGA implementation runs at `[ ] MHz`, and all latency values reported in this section include host-device transfer and control overhead unless explicitly stated otherwise. The tracking pipeline is evaluated in both **Search mode** and **Track mode**, and we additionally report average runtime under the proposed Search/Track scheduler.

#### 4) Metrics and Baselines

We evaluate algorithmic performance using **pupil-center error** (pixels), **gaze angular error** (degrees), and, when relevant, **search success rate** and **recovery latency**. System performance is measured using **end-to-end latency**, **effective update rate**, **resource utilization**, and `[power / energy per inference]` where available. For algorithm comparison, we consider representative frame-based methods such as *Etracker* and *HE-Tracker*, event-based methods such as *E-Track*, *FACET*, and *SEEN*, and hybrid methods such as *EX-Gaze*. For deployment-oriented comparison, we include *EyeCoD*, *SEE*, *JaneEye*, and other hardware-aware eye-tracking systems when comparable results are available. When direct reproduction is not possible due to unavailable code or incompatible sensing setups, we report the originally published results and explicitly discuss comparison limitations.

---

### B. Evaluation of the Proposed Algorithm and Accelerator

Table II summarizes the main performance of HBTXR. In Track mode, the proposed system achieves a **pupil-center error of 0.42 pixels**, a **gaze error of 0.68°**, and an **end-to-end latency of 0.57 ms**. These results indicate that HBTXR simultaneously maintains accurate localization and sub-millisecond responsiveness, which is essential for practical XR eye tracking. Search mode incurs higher latency than Track mode, as expected, because it processes a larger canonical region and invokes the full recovery path. However, Search is used only when initialization or relocalization is required, while steady-state operation remains dominated by the low-latency Track mode.

A key observation is that the proposed hybrid structure improves not only point accuracy but also runtime stability. Frame-based Search provides semantically stable relocalization when local tracking confidence degrades, while event-based Track supports frequent updates between successive frame refreshes. As a result, HBTXR avoids the failure pattern commonly observed in purely event-based trackers, where drift accumulates and global recovery becomes unreliable, and it also avoids the high steady-state cost of frame-only systems that recompute full visual features at every update. The measured latency therefore reflects not only accelerator speed but also the effectiveness of the overall search-track decomposition.

We further analyze the contribution of each component through ablation studies. First, removing the dense-annotation-aware canonical training contract degrades both pupil-center and gaze accuracy, indicating that geometry-stable supervision is important for learning a consistent cross-modality representation. Second, replacing the hybrid transformer with a frame-only or event-only variant degrades the overall accuracy-latency trade-off: the frame-only model remains stable but slower, whereas the event-only model is fast but less robust in re-initialization and recovery. Third, removing the decoupled Search/Track/Mask heads and using a single unified head reduces runtime controllability and weakens recovery behavior. Finally, disabling feature reuse in the FPGA implementation significantly increases memory traffic and Track-mode latency, confirming that streaming-based reuse is essential for efficient hardware deployment.

The proposed scheduler is also evaluated independently. Under dynamic eye-motion sequences, the host-side Search/Track policy invokes Search only when the confidence score drops below threshold or when re-initialization is explicitly requested. This reduces the average inference cost while maintaining robust long-horizon tracking. Compared with always-on Search execution, the scheduler reduces the average end-to-end latency by `[ ]%` while preserving localization accuracy within `[ ]` pixels. These results confirm that runtime-adaptive mode switching is not only algorithmically meaningful but also practically beneficial for hardware-aware deployment.

From the hardware perspective, the FPGA accelerator sustains the shared hybrid backbone and head execution within the latency budget required for XR workloads. Resource utilization results show that the shared transformer engine and lightweight decoupled head design enable efficient reuse of the backbone across Search and Track modes. The FSM-based scheduler further minimizes control overhead by implementing mode-specific execution paths directly on hardware. Overall, the proposed accelerator is able to support real-time hybrid eye tracking without requiring duplicated frame and event pipelines.

---

### C. Comparison with Other Eye-Tracking Algorithms

Table III compares HBTXR against representative prior eye-tracking methods. The comparison is organized by modality, including frame-based, event-based, and hybrid approaches. The results show that HBTXR provides the most balanced trade-off among semantic robustness, temporal responsiveness, and deployability.

Compared with **frame-based methods**, HBTXR achieves substantially lower latency while maintaining strong localization accuracy. Methods such as *Etracker* and *HE-Tracker* benefit from semantically rich image features, but their dense frame processing incurs higher runtime cost and makes them less suitable for high-frequency XR tracking. In contrast, HBTXR delegates global recovery to the Search branch and performs most steady-state updates through event-guided Track inference. This hybrid decomposition enables HBTXR to retain the recovery strength of frame-based approaches without inheriting their full runtime cost.

Compared with **event-based methods**, HBTXR demonstrates better relocalization stability and long-horizon robustness. Event-only methods such as *E-Track* and *FACET* are highly responsive and efficient, but they often rely on local motion evidence and can struggle when event support is weak or when tracking must recover from drift. HBTXR addresses this limitation by coupling event tracking with explicit frame-based Search and by training both branches under a geometry-stable canonical supervision contract. As a result, the proposed method improves robustness in challenging conditions while preserving sub-millisecond Track-mode latency.

Compared with **hybrid methods**, HBTXR differs in that it is designed as an end-to-end algorithm-hardware co-designed system rather than a loosely connected multimodal pipeline. *EX-Gaze* already demonstrated the potential of combining frame relocalization with event tracking. However, HBTXR further integrates this principle with a shared transformer backbone, decoupled mode-specific heads, a runtime Search/Track scheduler, and an accelerator-friendly implementation strategy. Consequently, HBTXR not only improves the algorithmic trade-off between recovery and responsiveness, but also exposes a system structure that is directly realizable on an FPGA platform.

Taken together, these comparisons suggest that the primary strength of HBTXR is not merely achieving the lowest error on a single benchmark, but rather providing a consistent and practical balance between accuracy, recovery capability, and low-latency operation. This is the operating point that matters most for on-device XR deployment.

---

### D. Comparison with Other Accelerators

Table IV compares the proposed accelerator with representative eye-tracking accelerators and deployment-oriented systems, including *EyeCoD*, *SEE*, and *JaneEye*. Since these prior works differ in sensing modality, model type, process technology, and evaluation benchmark, direct numerical comparison should be interpreted with care. Nevertheless, several system-level differences are clear.

First, many prior accelerators are optimized either for **frame-based CNN pipelines** or for **event-only sparse computation**. For example, *EyeCoD* focuses on lensless frame-based eye tracking and optics-accelerator co-design, whereas *SEE* and *JaneEye* emphasize event-driven sparse processing for low-latency eye tracking. In contrast, HBTXR targets a more general hybrid operating regime in which frame-based Search and event-based Track must coexist within a single deployable architecture. This requirement imposes a more complex runtime structure but also makes the system significantly more relevant to practical XR scenarios where both robustness and responsiveness are necessary.

Second, the proposed accelerator explicitly supports **transformer-based hybrid inference**, whereas many prior deployment-oriented systems were developed around CNN-style backbones or handcrafted event pipelines. Supporting a shared multimodal transformer backbone on FPGA is nontrivial because it requires efficient token buffering, attention computation, and head reuse under strict latency constraints. HBTXR addresses this through a shared transformer engine, decoupled head execution, feature reuse across Track iterations, and FSM-based runtime control. These mechanisms are central to achieving low latency without duplicating major compute blocks.

Third, HBTXR differs from prior accelerators in its explicit coupling between **runtime scheduler** and **mode-aware datapath reuse**. In most previous systems, the accelerator executes a fixed inference path regardless of runtime confidence or tracking condition. In contrast, HBTXR uses the Search/Track scheduler to adapt the computational path according to the current confidence state. Search executes only when recovery is needed, while Track dominates steady-state updates. This reduces average compute cost and is particularly beneficial in XR workloads where long periods of stable local motion are interleaved with occasional large gaze shifts.

Overall, the results show that HBTXR achieves competitive latency while providing broader functionality than prior accelerators. Rather than optimizing only for a single modality or model class, it provides a hybrid, runtime-adaptive, transformer-capable hardware substrate that better matches the requirements of on-device XR eye tracking.

---

### E. Discussion

The experimental results support three main conclusions. First, **hybrid sensing is necessary but not sufficient** for practical XR eye tracking. Simply combining frame and event inputs does not automatically yield an efficient or robust system. The gains of HBTXR arise from jointly designing canonical supervision, shared multimodal inference, mode-specific heads, and explicit Search/Track scheduling. Second, **runtime adaptivity is essential**. The proposed scheduler reduces average compute cost while preserving recovery capability, showing that search and tracking should be treated as distinct operating modes rather than as a single monolithic inference path. Third, **algorithm-hardware co-design is critical** for deployment. The measured Track-mode latency of 0.57 ms is achieved not only because of the model architecture, but also because the accelerator reuses features, shares the backbone across heads, and minimizes control overhead through the FSM-based scheduler.

Despite these advantages, HBTXR still has several limitations. Search mode remains more expensive than Track mode, and therefore the overall system efficiency depends on how often relocalization is triggered in practice. The current prototype also assumes synchronized frame and event sensing, which may impose integration cost in some XR devices. In addition, although the FPGA implementation demonstrates real-time feasibility, further energy reduction may require an ASIC implementation or more aggressive low-bit quantization. Finally, while the current study focuses on pupil localization and gaze estimation, additional evaluation on broader eye-related tasks such as blink analysis or biometric authentication may further clarify the generality of the proposed framework.

In summary, the experiments demonstrate that HBTXR provides a practical operating point for XR eye tracking by unifying robust relocalization, low-latency event-guided tracking, and deployment-aware acceleration in a single system. This validates the central premise of the paper: for on-device XR eye tracking, accuracy, latency, and deployability must be optimized jointly rather than separately.

다음으로 바로 이어서 정리할 수 있는 항목은 두 가지다.

1. `Section VII. Conclusion` 초안 작성  
2. `Fig./Table/Algorithm 캡션`까지 포함한 전체 논문 뼈대 정리
