# DiffCode: Visual-Diff-Grounded Cross-Format Code Refinement

> 最后更新: 2026-04-09 10:00 UTC | 更新原因: Phase 0 完成，进入 Phase 1 pilot 数据生成

## 研究目标

提出 DiffCode——基于显式多尺度视觉差异感知的跨格式代码精炼方法。核心贡献：
1. **Differential Perception Adapter (DPA)**：插入 Qwen2.5-VL-7B 视觉编码器的轻量模块（~12M），从 full-attention 层（7, 23, 31）提取多尺度 diff 特征，在 spatial merge 之前做特征相减
2. **跨格式训练 + 错误类型分类体系**：~20K HTML+SVG difference-aligned 训练对，统一 6 类错误分类
3. **细粒度迁移分析**：按错误类型分析 HTML↔SVG 迁移率

**目标会议**：AAAI 2027（abstract Jul 25, paper Aug 1）

## 当前阶段

✅ Phase 0: 代码基础设施 — 完成
🟡 Phase 1: Pilot 数据生成 + Exp-0 方向验证

## 关键约束

- GPU: 训练服务器待用户指定（A6000 仅用于代码开发/数据生成，不用于训练）
- 无 RL 训练，纯 SFT 隔离 DPA 贡献
- DPA 架构预定义（layers 7/23/31, 2 层 cross-attention, 1280 hidden dim）
- Qwen2.5-VL-7B + LoRA，单卡 48GB 可放下
- HTML 渲染 1280×720 vs SVG 256×256，评估时需统一尺寸

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
- [x] SVG 扰动管线（6 类扰动）
- [x] HTML 扰动管线（6 类扰动）
- [x] 统一错误分类标注（error_taxonomy.py）
- [x] DPA 模块 + unit test + ViT hook shape 修复
- [x] 评估指标模块（SSIM/CLIP-Score/CodeBLEU/pass rate）
- [x] Git + W&B 基础设施
- [ ] 生成 pilot 数据（1K HTML + 1K SVG）
- [ ] 确定训练服务器 + 下载 Qwen2.5-VL-7B
- [ ] 运行 Exp-0 pilot

## 进展摘要

- [2026-04-09] Phase 0 完成：SVG pipeline（643行，6类扰动）、HTML pipeline（666行，6类扰动）、DPA（~12.5M params，11 tests）、评估指标（MetricsComputer，14 tests）、ViT hook 2D→3D reshape 修复、Git+W&B 就绪
- [2026-04-09] Reviewer 审查 ROADMAP：采纳放弃 EMNLP、定义评估指标、增加 Exp-0 pilot、增加外部 baseline（D002）
- [2026-04-09] 3 个 Worker 因服务器重启同时失败，HTML pipeline 第三次成功

## 关键决策日志

| ID | 日期 | 摘要 | 关联实验 |
|----|------|------|----------|
| D004 | 2026-04-09 | Exp-0 pilot 恢复为完整 2K（1K HTML + 1K SVG），覆盖 D003 的 S | — |
| D003 | 2026-04-09 | Exp-0 pilot 改为 SVG-only ~1K 样本，不等 HTML pipeline | — |
| D002 | 2026-04-09 | 响应 Reviewer 审查：放弃 EMNLP 备选，增加评估指标体系、Exp-0 pilot 验证 | — |
| D001 | 2026-04-09 | 立项启动 DiffCode，Phase 0 聚焦数据管线和 DPA 模块实现 | — |

