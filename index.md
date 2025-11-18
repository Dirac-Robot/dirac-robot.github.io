# Profile
## **Jeonggeun Song**
**AI Research Engineer & ML Systems Architect | Applied AI Research Engineer - Architecting Models, Metrics, and Systems**
Research-first engineer who designs and implements new system patterns to drive novel ideas into production, from large-scale multimodal pipelines to full LLM alignment and reproducibility stacks.

- Location: Seoul, South Korea
- Email: js110182@gmail.com
- GitHub: https://github.com/Dirac-Robot
- LinkedIn: https://www.linkedin.com/in/jeonggeun-song-860749142
- Medium: https://medium.com/@js110182

---
---
# Introduction

I am an engineer and researcher who analyzes systems from first principles, maps the information flow, and rebuilds end-to-end stacks around more principled abstractions. I have repeatedly taken underperforming or fragile systems-from mobile-scale CV models to large-scale multimodal pipelines-and turned them into stable, inspectable, and reproducible architectures under real product constraints.

I am the first author of ViBid (UAI 2023) and the co-first author of SymBa (FF-style symmetric contrastive learning). My work ranges from AutoML, NAS, and compact vision models to modern LLM alignment and full-stack systems. I also built Ato, a thin operating layer for reproducible experimentation, which integrates structural hashing with code and runtime fingerprinting. Across projects, I have typically led both the research and engineering tracks, taking responsibility for the full path from hypothesis to deployed system.

My work consistently focuses on identifying structural bottlenecks and rebuilding systems around more principled abstractions.

At my core I am research-driven: I start from first-principles formulations and conceptual clarity, then push those ideas all the way into production-grade systems. Most of my engineering depth comes from **designing new system patterns to make novel research actually run at scale**-from billion-scale multimodal data engines to full LLM alignment stacks and lightweight reproducibility layers.

### Key Achievements
- Built an 8-node, 64-GPU data engine (1M image-text pairs/hour, 200M+ pairs total) for large-scale multimodal data generation.
- Architected an end-to-end LLM alignment stack (SFT + DPO, vLLM-based serving) achieving ≥10-turn stable persona interactions without drift.
- Created "Ato", a minimal reproducibility layer (structural hashing + code/runtime fingerprinting) that plugs into existing training stacks with near-zero friction.
- First author of ViBid (UAI 2023) and co-first author of SymBa, cited by Science, Nature Communications, etc.

### Core Capabilities:
- Model & Architecture Design (ViT variants, Linear Attention, Contrastive Objectives)
- Large-Scale Distributed Systems (MPI pipelines, GPU orchestration, async batching)
- Reproducibility & Experiment Frameworks (structural hashing, runtime fingerprinting)
- End-to-End LLM Stacks (SFT/DPO, inference serving, evaluation metrics)
- Tech: Python, PyTorch, MPI, vLLM, ONNX, Distributed Systems

### Core Experiences Timeline:
- **(2018-2021)** AutoML & Architecture Search  
    Hyperparameter optimization, early NAS research, and integrating AutoML workflows into internal MLOps systems.
- **(2019-2022)** Vision Optimization & Efficiency  
    Face detection/recognition, pose estimation, and deployment to constrained devices and real services.
- **(2022)** Large-Scale Multimodal & Distributed Systems  
    Designed and operated BodCa, an 8-node/64-GPU image-text pipeline for high-throughput multimodal data generation.
- **(2023-2025)** End-to-End Systems & Applied LLM Research  
    Developed **Ato**, a structural reproducibility layer; built full LLM stacks (data → SFT/DPO → inference → evaluation) and long-turn persona systems.

---
---
# PROJECTS

## **LLM Optimization & Alignment (2025.08-2025.10) - 8B Multi-Persona Conversational Engine**
**Role:** Lead Engineer / Researcher  
**Summary:** Developed a multi-persona LLM engine for AI K-POP performer personas, enabling distinct vocal / linguistic styles, emotional tone, and relational dynamics without retrieval or fine-tuning - purely via structured persona prompts.

### Responsibilities & Contributions
- Designed and executed the full **DPO workflow** for persona chat:
  - multi-answer sampling per turn,  
  - rejection sampling and ranking-by-summary,  
  - persona-aware prompt-chosen-rejected triplet generation.
- Introduced a **resampling-based preference loop**: when the model's reasoning about "who said what to whom" was correct but the surface style felt off, generated up to 6 alternatives and selected human-preferred variants, then fed these back into DPO to correct style drift without harming reasoning.
- Eliminated SFT-stage multi-turn collapse (3-turn limit, frequent per-turn errors) and achieved **≧10-turn stable persona and group interactions** with minimal drift after DPO tuning.
- Defined and tracked evaluation protocols for:
  - persona consistency and catchphrase retention,  
  - style and emotional tone adherence,  
  - error rate and incoherence across long conversations.
- Improved annotation throughput by **≧3×** via an internal UI:
  - one-click bad-answer rejection and resample,  
  - server-side preset storage for prompts/personas,  
  - "test mode" for safe experimentation on new prompts and models.
- Optimized inference latency to **sub-second responses** for typical persona turns through vLLM + Tensor Parallel tuning, caching strategies, and careful system-level profiling.

### Outcome / Impact
- Delivered a **production-grade 8B multi-persona LLM** capable of natural Korean small talk, naturalistic conversational tone, and multi-agent persona interactions without retrieval.
- Produced a reusable **alignment template** (DPO + resampling + evaluation protocol) for future persona/agent-style products.

---

## **End-to-End LLM Stack - Data, Training, Serving (2025.04-2025.08)**
**Role:** Lead Architect & Implementer (solo for core infra)  
**Summary:** Designed and implemented an internal LLM stack from scratch for persona-based chat systems, covering data, training, serving, and tooling - including a prompt-only multi-persona conditioning scheme on top of an 8B base model.

### System Design
- **Data Pipeline**
  - Built the full pipeline from raw chat logs and synthetic dialogs to **persona-conditioned training sets (~20K multi-turn dialog turns)**.
  - Implemented filtering, de-duplication, and safety checks to produce high-quality SFT and DPO datasets focused on everyday tone, subtle humor, and natural rhythm.
- **Persona Conditioning**
  - Designed structured **persona prompt templates** (backstory, role, tone, emotional traits, catchphrases) that allow the same 8B base model to adapt to **arbitrary unseen personas** without LoRA/MoE or architecture changes.
  - Enabled **multi-character group conversations**, where multiple personas respond to the user and to each other while preserving relationship dynamics of diverse persona archetypes (e.g., leader-like, shy, cheerful).
- **Training Pipeline**
  - Implemented training loops for SFT (single-turn / multi-turn) and DPO with persona-aware triplets.
  - Integrated logging, checkpointing, and metric tracking, tied into an Ato-style config / experiment management layer for local reproducibility.
- **Serving & Infra**
  - Deployed models with **vLLM-based serving**, including batching, streaming, and endpoint management.
  - Implemented routing logic for different personas and prompt templates, plus simple policy hooks for future safety/guardrail integration.

### Contributions
- Defined the end-to-end **persona prompting scheme** (system / style / safety structure) used across data generation, training, and serving.
- Built the internal "Watson" annotation & testing UI for iterative refinement of responses, personas, and prompts.
- Tuned performance (latency, throughput, VRAM usage) so that **8B persona chat** remained cost-efficient for product experiments.

### Outcome / Impact
- Delivered a fully functional **in-house LLM stack** suitable for persona chat experiments and early deployments.
- Established a robust technical foundation (data, training, serving, tooling) that future team members can extend rather than re-implement.

---
## **Controllable Motion Video Generation (2025.09-2025.10)
**Role:** Solo Engineer / Creator  
**Summary:** Built a controllable dance video generation pipeline on top of Alibaba's WAN 2.2 / WAN-Animate using ComfyUI, focusing on identity-preserving motion and inference-time stabilization without any fine-tuning.

### Responsibilities & Contributions
- Designed a **pose-driven video workflow**: extracted keypoints from source dance footage, controlled frame rate, resolution, and input quality to reduce pose jitter and temporal artifacts before feeding them into the diffusion pipeline.
- Implemented a **minimal-control strategy** to mitigate identity drift:
  - used an initial reference frame plus pose control only,  
  - disabled additional controls (e.g., face / edge) that degraded identity stability,
  - tuned reference-frame count, sampler type, step count, and denoise strength through empirical sweeps.
- Generated base character images with **Google  Nano Banana**, and used LLM-assisted prompt polishing plus end-frame conditioning to realize simple but precise motions (e.g., gestures, small choreography segments).
- Operated entirely at **inference level** (no fine-tuning), learning and exploiting WAN's inductive biases for motion and identity to achieve stable, reusable ComfyUI graphs for future dance/animation clips.

### Outcome / Impact
- Produced 30-60s dance clips with **consistent character identity and smooth motion**, including realistic, 2D illust, and 3D animation styles, gaining hands-on intuition about video diffusion failure modes (identity drift, flicker, pose distortion) and practical control strategies that transfer to larger-scale generative video systems.

---
## **Large-Scale Image-Text Pipeline "BodCa" (2022.05-2022.10)**
**Role:** Independent Architect & Sole Developer  
**Summary:** Designed and implemented a fully distributed 8-node/64-GPU image-text pipeline producing ~1M samples/hour and 200M+ curated pairs total.

### System Architecture
- Multi-node controller-worker design with MPI/queue-based orchestration.
- GPU workers handled decoding, augmentation, and encoding; CPU workers managed I/O, scheduling, and metadata.
- Integrated model-based filtering and scoring to maintain data quality.

### Key Contributions
- Achieved ~1M image-text pairs/hour throughput under realistic hardware constraints.
- Designed modular stages for decoding, cropping, captioning, and filtering, allowing plug-and-play replacement of models.
- Implemented robust failure handling, resumption, and logging suitable for long-running distributed jobs.

### Outcome / Impact
- Provided a scalable data engine for multimodal pretraining, similar in spirit to ALIGN/CLIP-style pipelines.
- Demonstrated ability to own a complex distributed system end-to-end (design, implementation, operation).

---
## **Traffic Light Detection (TLD) - DINO-based Object Detection (2024-2025)**
**Role:** Lead Engineer for Optimization & Deployment
**Summary:** Rebuilt company's entire TLD system from scratch: dataset, architecture, training pipeline, evaluation, and deployment. Improved Pass Rate from ~30% to 70%+ under strict autonomous driving safety criteria.

### Key Contributions
- Replaced Hydra/MMDetection stack with custom Ato-based experimentation framework.
- Benchmarked multiple architectures; deployed optimized DINO (DETR-family) detector.
- Fixed DETR duplicate-box issues with custom NMS, improving dense-scene precision.
- Converted unstable multi-head classifier into a stable single-head formulation.
- Filtered and curated data using environmental metadata; built targeted augmentation sets.
- Integrated model into annotation → validation → deployment cycle.

### Outcome / Impact
- Increased Pass Rate from ~30% → 70%+ under strict failure criteria.
- Delivered stable, continuously improving TLD model supporting entire AD pipeline.

---
## **Face Recognition & Verification Systems (2019-2021)**
**Role:** Vision Engineer / Lead Implementer  
**Summary:** Built and optimized face recognition and verification systems for real-world applications, focusing on robustness and efficiency.

### Contributions
- Implemented and tuned backbone architectures for face recognition.
- Stabilized training by diagnosing and fixing NaN issues in a legacy custom ArcFace loss implementation, improving convergence and deployment reliability.
- Designed data pipelines for large-scale face datasets, including cleaning, augmentation, and hard-example mining.
- Deployed models to production with attention to latency and resource constraints.

### Outcome / Impact
- Delivered robust face recognition pipelines suitable for production use.
- Built foundation for later efficiency and architecture work in vision.

---
---
# PERSONAL DEV PROJECTS

## **A Thin Operating Layer "Ato" (2023-2025)**
**Role:** Sole Author & Architect  
**Summary:** Built a modular reproducibility layer capturing configuration structure and runtime behavior via structural hashing, code fingerprinting, and SQL-based lineage tracking.

### Key Contributions
- Structural-hash config engine (ADict), causal merge system (Scope), MultiScope isolation.
- Deterministic bytecode-level code versioning; runtime fingerprinting for nondeterminism.
- SQLite-backed tracker supporting similarity search, lineage queries, trace statistics.
- ~100 deterministic tests across ~10 files; zero-coupling with any ML stack.

### Outcome / Impact
- Practical reproducibility layer complementing Hydra/MLflow/W&B without heavy coupling.

---
---
# RESEARCH

### My research focuses on structural learning problems - e.g., deriving principled views of diffusion as real-distribution estimators, designing BP-free contrastive objectives, and building architectures/metrics that align clean theory with deployable systems.

## **Meta-Context Diffusion — Context Refinement Module (2024)**
**Role:** Co-first Author, Sole Contributor
**Summary:** Proposed Meta-Context Generator enabling diffusion models to infer implicit context for ambiguous prompts.

### Selected Contributions
- Defined meta-context space via variational reparameterization.
- Implemented cross-attention plugin compatible with SDXL/SD3.
- Demonstrated context disentanglement and fidelity improvements (red lips, hands).
- Trained with only 8k COCO captions in 2–4h.

### ### Impact
- Enables diffusion models to **resolve prompt–image ambiguity** by injecting an inferred meta-context that removes the boundary between what is explicitly specified in the prompt and what must be implicitly reconstructed.
- Provides a **generalizable refinement layer** with negligible runtime overhead and plug-and-play compatibility (SDXL/SD3), improving semantic fidelity in challenging attributes (e.g., mouth, lips, hands) without modifying the base model.
- Achieves measurable gains in **domain-agnostic context disentanglement and attribute precision** using only 8k COCO captions trained within 2–4 hours, demonstrating high data/compute efficiency.

---
## **SymBa - Symmetric BP-Free Contrastive Learning (2023)**
**Role:** Co-first Author  
**Summary:** Introduced symmetric goodness-gap contrastive formulation improving FF-style learning stability and accuracy.

### Selected Contributions
- Proposed symmetric contrastive objective tailored to FF-style learning without backprop.
- Analyzed training dynamics and stability under the new objective.
- Ran extensive experiments across multiple datasets and architectures.

### Impact
- Although not formally accepted, but later cited by high-impact venues including **Science, Nature Communications, Pattern Recognition, Neural Networks, Computer Vision and Image Understanding (CVIU), IEEE WF-IoT, IEEE COINS, and ICLR**.
- Link: https://arxiv.org/abs/2303.08418

---
## **ViBid - Linear Vision Transformer with BiNorm (UAI 2023)**
**Role:** First Author  
**Summary:** Developed BiNorm-based O(N) transformer architecture with strong GPU efficiency and competitive ImageNet accuracy.

### Technical Contributions
- Invented BiNorm dual-normalization resolving Softmax-free instability.
- Introduced Q(KᵀV) attention ordering for true O(N) complexity.
- Built full ViBid architecture: patch embedding, BiNorm attention, locality modules.
- Led 400-epoch ImageNet training & high-res fine-tuning.

### Impact
- Clean, principled O(N) attention with practical GPU advantages.
- Published at UAI 2023 (PMLR).  
- Link: https://proceedings.mlr.press/v216/song23a.html

---
---
# Careers

- 2025.04 - Present | Blue Garage
- 2024.01 - 2025.03 | 42Dot
- 2023.08 - 2024.01 | Kakao Brain
- 2021.09 - 2023.08 | Kakao Enterprise
- 2018.08 - 2021.08 | Kakao

---
---
# Skills

- Domains: LLM alignment (SFT/DPO, persona systems), large-scale multimodal data pipelines, computer vision (detection, recognition, NAS), reproducibility frameworks.
- Frameworks & Tools: PyTorch, vLLM, ONNX, MPI
- Data & Systems: Distributed GPU pipelines (8-node/64-GPU), async batching, SQL/SQLite-based experiment tracking and lineage analysis.
- Research & Methods: Contrastive learning, linear/efficient attention, distillation, neural architecture search, evaluation and metrics design.
 
---
---
