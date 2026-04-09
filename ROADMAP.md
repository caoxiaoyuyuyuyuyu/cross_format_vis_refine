# DiffCode: Visual-Diff-Grounded Cross-Format Code Refinement

> 最后更新: 2026-04-09 14:40 UTC | 更新原因: 服务器部署完成，更新硬件描述和短期任务

## 研究目标

提出 DiffCode——基于显式多尺度视觉差异感知的跨格式代码精炼方法。核心贡献：
1. **Differential Perception Adapter (DPA)**：插入 Qwen2.5-VL-7B 视觉编码器的轻量模块（~12M），从 full-attention 层（7, 23, 31）提取多尺度 diff 特征，在 spatial merge 之前做特征相减
2. **跨格式训练 + 错误类型分类体系**：~20K HTML+SVG difference-aligned 训练对，统一 6 类错误分类
3. **细粒度迁移分析**：按错误类型分析 HTML↔SVG 迁移率

**目标会议**：AAAI 2027（abstract Jul 25, paper Aug 1）

## 当前阶段

✅ Phase 0: 代码基础设施 — 完成
✅ Phase 1a: 服务器部署 + 环境搭建 — 完成
🟡 Phase 1b: Pilot 数据生成 + Exp-0 方向验证

## 关键约束

- **训练服务器**：`autodl-cross-vis`（1× NVIDIA RTX PRO 6000 Blackwell 98GB，torch 2.10.0+cu128）
- 单卡 98GB 足够 Exp-0（7B+LoRA），20K 全量训练时评估是否需多卡
- 无 RL 训练，纯 SFT 隔离 DPA 贡献
- DPA 架构预定义（layers 7/23/31, 2 层 cross-attention, 1280 hidden dim）
- HTML 渲染 1280×720 vs SVG 256×256，评估时需统一尺寸
- **Blackwell sm_120 必须 cu128+**（cu126 及以下不兼容）

## 评估指标体系

**视觉相似度**（主指标）：SSIM + CLIP-Score
**代码级**（辅助）：CodeBLEU
**精炼效果**：Per-error-type Pass Rate（SSIM > 0.95）、Multi-round Improvement Curve、Cross-format Transfer Rate
**注意**：SSIM 用灰度计算，color 类可能不敏感，pilot 后校准

## 实验计划

### Exp-0: Pilot 验证（~2K 样本）⬅️ 当前
1K HTML + 1K SVG，Baseline vs DiffCode。Gate: SSIM 提升 ≥1%。~2h GPU。

### Exp-1: Baseline（隐式比较，20K joint）
### Exp-2: DiffCode（DPA + 显式 diff tokens，20K joint）
### Exp-3: 跨格式迁移（per-error-type breakdown）
### Exp-4: 层消融（full-attention vs 混合）
### Exp-5: 外部 Baseline 对比（VisRefiner 复现 + zero-shot）

## 短期任务

- [x] 项目仓库搭建
- [x] SVG/HTML 扰动管线 + 统一错误分类
- [x] DPA 模块 + 评估指标模块
- [x] Git + W&B 基础设施
- [x] 训练服务器部署（autodl-cross-vis, torch cu128, 模型就绪）
- [ ] 生成 pilot 数据（1K HTML + 1K SVG）
- [ ] 运行 Exp-0 pilot

## 进展摘要

- [2026-04-09] Phase 0 完成：SVG/HTML pipeline、DPA（~12.5M params）、评估指标、ViT hook 修复
- [2026-04-09] 服务器部署：autodl-search-traj 因网络超时+cusparseLt缺失反复失败 → 用户换服务器 → autodl-cross-vis 部署成功（torch 2.10.0+cu128, Blackwell sm_120 兼容）
- [2026-04-09] 关键教训：Blackwell GPU sm_120 需要 cu128+，cu126 kernel 不兼容；AutoDL 系统盘空间小，venv 须放数据盘

## 关键决策日志

| ID | 日期 | 摘要 | 关联实验 |
|----|------|------|----------|
| D006 | 2026-04-09 | Exp-0 增加 prompt 难度消融：easy(+error_desc) vs hard(-er | — |
| D005 | 2026-04-09 | 统一 attn_implementation 为 sdpa，取消 flash-attn 依赖 | — |
| D004 | 2026-04-09 | Exp-0 pilot 恢复为完整 2K（1K HTML + 1K SVG），覆盖 D003 的 S | — |
| D003 | 2026-04-09 | Exp-0 pilot 改为 SVG-only ~1K 样本，不等 HTML pipeline | — |
| D002 | 2026-04-09 | 响应 Reviewer 审查：放弃 EMNLP 备选，增加评估指标体系、Exp-0 pilot 验证 | — |
| D001 | 2026-04-09 | 立项启动 DiffCode，Phase 0 聚焦数据管线和 DPA 模块实现 | — |

