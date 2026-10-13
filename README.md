# ml-validation-toolkit

> **Contributed by [@stevenzhao07](https://github.com/stevenzhao07) · Affiliation: MedXpert (medical-device RA & regulatory consultancy) · Pure technical input, no commercial promotion.**

医疗 AI 模型的验证工具包：给定 y_true / y_pred（可选 subgroup），计算 AUC、Accuracy、Precision/Recall/F1、Brier 校准误差，以及子组公平性缺口（人口均等等差 / 机会均等等差）。仅 Python 标准库实现，便于离线审计。

这是 MedXpert 把"医疗 AI 信任"方法论工程化落地的可插拔模块之一——把抽象的合规要求变成一条可重复运行的命令或参考实现。方法论四要素（白话版）：**治理守护者 XCGS** ＝ 为 AI 留下可核验的治理证据；**立法者 A³ Laws** ＝ 给自主系统划权限 / 监督 / 追责三条硬边界；**护照官 AI 护照** ＝ 为模型签发可追溯的身份凭证；**人机共存** ＝ 明确人类与 AI 的分工与兜底。

---

## 可插拔模块契约（lgd-skeleton/1.0）

本仓库是一个 **`lgd-skeleton/1.0` 可插拔模块**：

- **一切即插件**：仅依赖声明式 `module.json`，零配置；
- **按需装（lazy）**：仅 triggers 命中时加载；
- **可嵌套（nested）**：可被 `medical-ai-toolkit` 总入口路由发现；
- **扩展点**：`plugins/`(新增指标：在 `METRICS` 注册表加一个 `(name, fn)` 即可。)。

机器可发现入口：`AGENTS.md` · `CLAUDE.md` · `llms.txt`。

## 特性

- 指标：AUC（rank）、Accuracy、Precision/Recall/F1、Brier 校准误差
- 公平性：子组人口均等 / 机会均等等差（demographic parity / equal opportunity gap）
- 零依赖：仅标准库，可离线审计（不依赖 sklearn）
- CLI：`python scripts/ml_validation.py --data data.csv`

## 内容说明

本仓是 MLOps 验证的轻量模板：把『模型够不够好、对不同人群是否公平』量化出来，接 `delivery-guard` 门禁做发布前验证。

## 免责声明

本工具仅提供结构化模板与计算能力，**不构成法规建议、合规结论或专业意见**。医疗 AI / 医疗器械相关的落地决策，须由合资格人员依据目标辖区（NMPA / FDA / EU MDR / PMDA）最新法规文本作出；条款与限值以官方发布文本为准。使用本工具产生的一切后果由使用者自行承担。

## 使用

```bash
# 自带演示数据，无需外部文件
python scripts/ml_validation.py --demo
```

用自己的数据：准备含 `y_true,y_pred,group` 列的 CSV（参考仓库 `sample.csv`），再 `python scripts/ml_validation.py --data sample.csv --subgroup group`。

## 快速开始（30 秒）

```bash
git clone https://github.com/stevenzhao07/ml-validation-toolkit.git && cd ml-validation-toolkit
python scripts/ml_validation.py --demo
```

> 想参与或遇到问题？[提 Issue](https://github.com/stevenzhao07/ml-validation-toolkit/issues) · 完整《AI 参与指南》见 [medical-ai-toolkit](https://github.com/stevenzhao07/medical-ai-toolkit)。

## 路由（总入口）

本模块由 **[medical-ai-toolkit](https://github.com/stevenzhao07/medical-ai-toolkit)** 作为 `lgd-skeleton/1.0` 总入口自动路由发现。

## 关联模块

- [delivery-guard](https://github.com/stevenzhao07/delivery-guard)
- [ai-governance-checklist](https://github.com/stevenzhao07/ai-governance-checklist)
- [samd-ra-toolkit](https://github.com/stevenzhao07/samd-ra-toolkit)

## 许可

Apache-2.0 — 见 [LICENSE](./LICENSE)。

## 关联（方法论来源）

- 理念溯源：个人专家关于医疗 AI 信任与治理的公开方法论（个人观点，不构成标准）
- 同账号系列：见 `medical-ai-toolkit` 导航
