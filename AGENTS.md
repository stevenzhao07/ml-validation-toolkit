# AGENTS.md · ml-validation-toolkit

> 机器可发现入口（AI 编码代理 / Codex / Claude / Cursor）。单一事实源：本文件。

## 这是什么
医疗 AI 模型的验证工具包：给定 y_true / y_pred（可选 subgroup），计算 AUC、Accuracy、Precision/Recall/F1、Brier 校准误差，以及子组公平性缺口（人口均等等差 / 机会均等等差）。仅 Python 标准库实现，便于离线审计。

## 运行
```bash
python scripts/ml_validation.py --demo
python scripts/ml_validation.py --help
```

## 扩展契约（lgd-skeleton/1.0）
- **一切即插件**：仅依赖声明式 `module.json`，零配置。
- **按需装（lazy）**：仅 triggers 命中时加载。
- **可嵌套（nested）**：可被 `medical-ai-toolkit` 总入口路由。
- **扩展点**：`plugins/`(新增指标：在 `METRICS` 注册表加一个 `(name, fn)` 即可。)

## 红线
纯技术贡献，不含商业营销；MedXpert 仅作为贡献者署名（@stevenzhao07）。
