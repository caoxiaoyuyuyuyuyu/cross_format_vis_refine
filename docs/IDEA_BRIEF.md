# Idea Brief: DiffCode: Visual-Diff-Grounded Cross-Format Code Refinement via Multi-Scale Differential Perception


- **Idea ID**: `idea_025`
- **Created**: 2026-04-08
- **Approved**: 2026-04-09
- **Source**: web_research
- **Level**: top-tier
- **Target Venues**: AAAI 2027 (abstract Jul 25, paper Aug 1), EMNLP 2026 ARR (backup, May 25 — if emphasizing code/language aspects)

## Seed Topic

帮我生成和LLM agent、harness engineering、LLM+搜索、LLM+graphics相关的idea，要求能中顶会，用双卡RTX PRO 6000（单卡96GB显存）及以下的资源（GPU 100h以下），尽量不用API key，不自己构建benchmark，不需要人工标注，尽量不要做RL训练（显存占用过高）！

## Generation Journey

1. **Start** — 开始生成，seed: '帮我生成和LLM agent、harness engineering、LLM+搜索、LLM+graphics相关的idea，要求能中顶会，用双卡RTX PRO 6000（单卡96GB显存）及以下的资源（GPU 100h以下），尽量不用API key，不自己构建benchmark，不需要人工标注，尽量不要做RL训练（显存占用过高）！'
2. **Problem Discovery** (714s) — 发现 4 个研究方向
   - SFT Distillation of Interleaved Search-Reasoning for Small LLMs
   - Rendering-Grounded Rejection Sampling for Visual Code SFT
   - Learned Online Trajectory Compression for LLM Agents
   - Multi-Format Visual Code Refinement via Difference-Aligned SFT
3. **Method Synthesis** (281s) — 完成: "SFT Distillation of Interleaved Search-Reasoning for Small LLMs" [solid] → NeurIPS 2026 (abstract May 4, full paper May 6), EMNLP 2026 (ARR submission May 25)
3. **Method Synthesis** (182s) — 完成: "Rendering-Grounded Rejection Sampling for Visual Code SFT" [solid] → NeurIPS 2026 (paper deadline May 6, 2026), EMNLP 2026 (ARR deadline May 25, 2026)
3. **Method Synthesis** (393s) — 完成: "Learned Online Trajectory Compression for LLM Agents" [solid] → NeurIPS 2026 (abstract May 4, paper May 6, 2026), EMNLP 2026 (ARR submission May 25, 2026; commitment Aug 2, 2026)
3. **Method Synthesis** (426s) — 完成: "Multi-Format Visual Code Refinement via Difference-Aligned SFT" [solid] → NeurIPS 2026 (abstract May 4, paper May 6), EMNLP 2026 (ARR submission May 25)

## Core Problem

Visual code refinement — iteratively fixing code by comparing rendered output against a target — works well within single formats (VisRefiner for HTML, ChartIR for charts, Amazon's vision-guided frontend refinement) but remains format-specific and relies on the VLM's implicit visual comparison ability during inference. Two problems are intertwined: (1) per-format pipelines are expensive to build and maintain, and (2) the visual comparison at inference time is implicit — the VLM receives two images (rendered output + target) and must internally figure out WHERE differences exist and WHAT TYPE they are, without any explicit differential features. Meanwhile, the image difference captioning community (OneDiff ACCV 2024, M3Diff/OmniDiff ICCV 2025) has developed multi-scale differential perception modules that produce structured, format-agnostic visual diff features — but these are used for describing differences in natural language, never for driving code fixes. Bridging structured visual diff perception into code refinement would provide the VLM with explicit 'what is different where' signals during iterative correction, enabling both more precise refinement and cross-format generalization.

## Opportunity / Why Now

VisRefiner (Feb 2026) proved that difference-aligned SFT on 20K (buggy_code, fixed_code) pairs enables strong iterative refinement at 7B scale, but during inference, the model implicitly compares two images without explicit differential features — it must re-discover visual differences at every refinement step. Amazon's concurrent vision-guided refinement (2026) achieves +17.8% over 3 cycles for HTML/CSS with a similar implicit-comparison approach. M3Diff (OmniDiff, ICCV 2025) introduced a Multi-scale Differential Perception (MDP) module that computes structured diff features via element-wise subtraction across multiple vision encoder layers, fused through cross-attention — achieving SOTA on image difference captioning across 12 visual change types. OneDiff (ACCV 2024) showed a Visual Delta Module can extract fine-grained difference features between image pairs. Neither has been applied to code refinement. The opportunity: design a visual-diff-grounded refinement method that (1) provides explicit multi-scale differential features to the LLM during inference-time iterative refinement, (2) maps visual error types to code fix actions via a unified taxonomy, and (3) works across HTML and SVG without format-specific training — filling the unoccupied niche of SVG iterative refinement with visual feedback.

## Landscape (SOTA & Limitations)

VisRefiner (Feb 2026): SFT on 20K difference-aligned (buggy_code, fixed_code) pairs achieves 90.8 on VisDiffUI-Test (92.0 with GRPO RL using CLIP-based RIS reward); during inference, model receives two images + code but performs implicit visual comparison; code NOT open-sourced. Amazon Vision-Guided Refinement (2026): VLM as visual critic for HTML/CSS, +17.8% over 3 refinement cycles; implicit comparison, no explicit diff features. VinciCoder (Nov 2025): 1.6M image-code pairs, 7 formats, ViRL with coarse-to-fine visual reward; SOTA on generation but single-pass. M3Diff/OmniDiff (ICCV 2025): MDP module with 12-category error taxonomy; uses SigLIP (27 layers, global attention throughout); SOTA on image difference captioning. OneDiff (ACCV 2024): Visual Delta Module with siamese encoder + cross-attention; +97% CIDEr improvement. VFLM (Mar 2026): visual feedback for iterative text layout refinement; similar self-improving loop but no structured diff features. UI2CodeN: iterative polishing for HTML, +12% over 4 rounds, 80K SFT samples. ChartIR: training-free iterative chart refinement, +75% quality via structured visual diffs. ScreenCoder (Jul 2025): modular multi-agent for visual-to-code, SOTA layout accuracy. CTRL-S: multi-task multi-reward RL for SVG, positive cross-task transfer. VisCodex (ICLR 2026): unified multimodal code benchmark. OmniSVG/MMSVG-2M (NeurIPS 2025): 2M SVG assets on HuggingFace. SVGEditBench V2: SVG editing benchmark (single-turn). No existing work combines explicit structured visual diff perception with code refinement, or evaluates cross-format transfer of visual debugging skills with fine-grained error-type analysis.

## Proposed Approach

We propose DiffCode, a visual-diff-grounded refinement method with three contributions:

**1. Differential Perception Adapter for Code Refinement (Method Contribution)**
Inspired by M3Diff's MDP module, we design a lightweight Differential Perception Adapter (DPA) that plugs into Qwen2.5-VL-7B's vision encoder. Key architectural decision: Qwen2.5-VL's ViT has 32 layers with full (global) attention only at layers [7, 15, 23, 31] — the remaining 28 layers use window attention (window_size=112) with limited receptive field. We tap the three upper full-attention layers (7, 23, 31) for multi-scale diff features, because: (a) full-attention layers have global receptive field, making element-wise feature subtraction semantically meaningful (window-attention features are spatially fragmented), (b) these three layers provide low-level (layer 7), mid-level (layer 23), and high-level (layer 31) representations, mirroring M3Diff's multi-scale design. Critically, Qwen2.5-VL applies spatial_merge_size=2 after the ViT, merging 2×2 patch blocks into single tokens — we perform feature subtraction BEFORE this spatial merge to preserve full spatial resolution for diff computation. The DPA pipeline: (a) extract features from layers 7, 23, 31 for both rendered and target images, (b) element-wise subtraction at each scale, (c) cross-attention fusion of diff features with original target features (2 cross-attention layers with 1280 hidden dim), (d) projection to LLM input space. The fused diff tokens are prepended to the LLM input alongside current code, providing explicit 'what is different where' signals. The DPA adds ~12M parameters and is trained end-to-end with LoRA. Architecture is pre-committed — no architecture search.

**2. Cross-Format Training with Error-Type Taxonomy (Empirical Contribution)**
We construct ~20K difference-aligned training pairs across HTML and SVG. For HTML (~10K): perturbation pipeline following VisRefiner's 6 categories (color, layout, alignment, component, image, text) using WebCode2M seeds rendered via headless Chrome. For SVG (~10K): analogous pipeline using MMSVG-Icon seeds (simpler SVGs per prior reviewer advice) with 4 high-signal perturbation types — fill/stroke color changes, coordinate offsets (5-20px), size scaling (0.7-1.3×), element removal — rendered via CairoSVG. Each training pair is annotated with error type labels from a unified 6-category taxonomy (color, position, size, element, text, style) that maps across both formats. For SVG evaluation, we construct a dedicated SVG iterative refinement test set (~500 samples) using held-out MMSVG-Icon seeds with multi-step perturbation chains, measuring per-round and per-error-type refinement success — filling a benchmark gap since no established iterative SVG refinement benchmark exists.

**3. Fine-Grained Transfer Analysis (Analysis Contribution)**
Beyond binary 'does transfer work?', we measure per-error-type transfer rates: train on HTML, test SVG refinement broken down by error type (and vice versa). Hypothesis: perceptual skills (color matching, size comparison) transfer well because they are format-agnostic visual operations, while structural skills (layout positioning) transfer poorly because HTML box model vs SVG coordinate system are fundamentally different. This turns any result — positive, negative, or mixed — into a rich empirical contribution.

**Key experiments**: (a) DPA vs implicit-comparison baseline: at inference time, does providing explicit diff tokens improve refinement over the standard two-image input that VisRefiner and Amazon's method use? (b) Cross-format transfer with per-error-type breakdown. (c) Joint training vs format-specific: does the DPA's format-agnostic diff representation enable positive transfer that implicit comparison does not? (d) Ablation: full-attention layers only (7, 23, 31) vs mixed (including window-attention layers) to validate our architecture choice. All SFT-only — no RL stage, deliberately isolating the contribution of explicit visual diff representation quality.

## Resource Estimate

- **auto_research**: {'claude_api_cost': '$200-360 (~2800-4500 turns)', 'estimated_gpu_cost': '$120-160', 'gpu_hours': '~80h A100 80GB (pre-committed DPA architecture + cross-format ablations + error-type analysis)', 'gpu_utilization': 'High (~80%). No architecture search — agent runs: DPA training → baseline comparison → transfer ablations → layer ablation.', 'risk_note': 'Pre-committed DPA architecture (layers 7, 23, 31; 2 cross-attention fusion layers) eliminates search cost. Budget allows 4 main training runs + 1 layer ablation + comprehensive error-type evaluation. ~15h buffer under 100h ceiling.', 'speed_bottleneck': 'HTML rendering via headless Chrome (~20 samples/min) is slower than SVG via CairoSVG (~100 samples/min). Total data generation: ~6-8 hours for 20K training + 500 SVG test samples.', 'team': '0 (only initial setup)', 'timeline': '1.5-2 weeks with dual RTX PRO 6000'}
- **human_in_loop**: {'claude_api_cost': '$120-220 (~1800-3000 turns)', 'estimated_gpu_cost': '$110-150', 'gpu_hours': '~75h A100 80GB', 'gpu_utilization': 'Moderate (~55%). Agent builds rendering pipelines; human reviews data quality.', 'risk_note': 'DPA integration with Qwen2.5-VL requires hooking into ViT forward pass at full-attention layers and extracting pre-spatial-merge features. MMSVG-Icon SVGs are simpler and CairoSVG handles them well. HTML pipeline built from scratch (~3-4 days). SVG eval test set construction ~1 day.', 'team': '1 researcher, ~25min/day check-in', 'timeline': '3-4 weeks'}
- **manual**: {'estimated_gpu_cost': '$95-130', 'gpu_hours': '~65h A100 80GB (or equivalent on dual RTX PRO 6000)', 'gpu_utilization': 'Low-moderate (~30%). GPU idles during rendering pipeline development.', 'team': '1 researcher', 'timeline': '6-8 weeks'}

## Key References

- VisRefiner: Learning from Visual Differences for Screenshot-to-Code Generation (arXiv Feb 2026) — SFT on 20K difference-aligned code pairs achieves 90.8; GRPO RL with CLIP-based RIS reward reaches 92.0; implicit visual comparison during inference; code NOT open-sourced
- M3Diff/OmniDiff: A Comprehensive Benchmark for Fine-grained Image Difference Captioning (ICCV 2025) — MDP module on SigLIP (27 layers, global attention); 12-category error taxonomy; key architectural inspiration for DPA; note SigLIP's global attention vs Qwen2.5-VL's window attention difference
- OneDiff: A Generalist Model for Image Difference Captioning (ACCV 2024) — Visual Delta Module with siamese encoder; format-agnostic visual diff features; +97% CIDEr improvement
- Amazon Vision-Guided Iterative Refinement for Frontend Code Generation (2026) — VLM as visual critic for HTML/CSS; +17.8% over 3 refinement cycles; concurrent work using implicit visual comparison
- VFLM: Visual Feedback for Iterative Text Layout Refinement (arXiv Mar 2026) — visual feedback self-improving loop for layout; concurrent work in visual iterative refinement paradigm
- VinciCoder: Unifying Multimodal Code Generation via Coarse-to-fine Visual Reinforcement Learning (arXiv Nov 2025) — 1.6M pairs, 7 formats, ViRL; single-pass comparison baseline
- OmniSVG/MMSVG-2M: A Unified Scalable Vector Graphics Generation Model (NeurIPS 2025) — 2M SVG dataset; source of MMSVG-Icon seeds for SVG pipeline
- VisCodex: A Unified Multimodal Code Benchmark (ICLR 2026) — unified benchmark across visual code formats; contextualizes cross-format evaluation
- CTRL-S: Reliable Reasoning in SVG-LLMs via Multi-Task Multi-Reward RL (arXiv Mar 2026) — evidence for cross-task transfer within SVG
- ChartIR: Training-Free Iterative Refinement for Chart-to-Code Generation (arXiv Jun 2025) — training-free iterative refinement via structured visual diffs
- ScreenCoder: Advancing Visual-to-Code Generation via Modular Multimodal Agents (arXiv Jul 2025) — modular grounding→planning→generation; SOTA layout accuracy
- UI2CodeN: Interactive UI-to-Code with Test-Time Scaling (arXiv Nov 2025) — iterative polishing for HTML; +12% over 4 rounds
- SVGEditBench V2: A Benchmark for Instruction-based SVG Editing (arXiv Feb 2025) — SVG editing benchmark (single-turn)
- Design2Code: Benchmarking Multimodal Code Generation for Automated Front-End Engineering (NAACL 2025) — HTML evaluation benchmark

## Reviewer Evaluation

**Scores**:
  - clarity: **{'rationale': 'Well-structured three-contribution framework (DPA method, cross-format training with taxonomy, fine-grained transfer analysis). Architecture decisions are justified (full-attention layers only, pre-spatial-merge subtraction). Experimental plan is concrete: 4 main experiments + 1 ablation, all pre-specified.', 'score': 4}**/5
  - feasibility: **{'rationale': 'Pre-committed DPA architecture eliminates search cost. LoRA on 7B + 12M DPA params fits dual RTX PRO 6000. MMSVG-Icon on HuggingFace is available. 80h auto estimate with 15h buffer under 100h ceiling is realistic. Main risk is HTML pipeline construction without VisRefiner code, but perturbation-render-pair approach is straightforward to implement independently.', 'score': 4}**/5
  - impact: **{'rationale': "Visual code generation is a growing subfield (VisRefiner, VinciCoder, Amazon, VFLM, ChartIR, ScreenCoder — all 2025-2026). This paper provides three 'firsts': first explicit diff-feature method for code refinement, first per-error-type cross-format transfer study, and first SVG iterative refinement benchmark. Practical value in replacing per-format pipelines with a single model. Suitable for AAAI's multimodal/application focus.", 'score': 4}**/5
  - novelty: **{'rationale': "Genuinely novel bridge: no prior work applies learned multi-scale visual diff features (from image difference captioning) to drive code refinement. The DPA is architecturally close to M3Diff's MDP but the application domain and integration with Qwen2.5-VL's full-attention layers are new. The per-error-type cross-format transfer analysis adds analytical novelty.", 'score': 4}**/5
