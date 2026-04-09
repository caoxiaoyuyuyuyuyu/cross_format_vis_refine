# Memory

## Dry-run 结果 (2026-04-09)

### Baseline (sdpa)
- Loss: 0.0805, trainable 5M/8.3B (0.06%), 1000 samples (train 800 / val 200)

### DiffCode (sdpa)
- Loss: 0.2949 (首步，高于 baseline 正常——diff tokens 前缀改变输入分布)
- DPA 参数量: 12.5M, 总 trainable: 17.5M/8.29B (0.21%)
- 注意力实现: sdpa

### 额外发现的 Bug（Worker 已修复在服务器上）
- Qwen2.5-VL 的 LLM 部分嵌套在 `.model.language_model` 下，非直接 `.model`
  - `base_model.model.embed_tokens` → `base_model.model.language_model.embed_tokens`
  - `base_model.model(...)` → `base_model.model.language_model(...)`
- 服务器无外网，模型路径必须用本地路径 `--model_name /root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct`

## HTML Pilot 数据
- 生成进行中（PID 19686），Worker 监控中

## 关键约束
- Blackwell GPU sm_120 需要 cu128+
- AutoDL 系统盘空间小，venv 须放数据盘
- attn_implementation 统一用 sdpa（D005 决策，不用 flash-attn）
