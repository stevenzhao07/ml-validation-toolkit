#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ml_validation.py — 医疗 AI 模型验证包（零依赖）
===============================================
给定 y_true / y_pred（可选 subgroup），计算：
  AUC(rank) / Accuracy / Precision / Recall / F1 / Brier 校准
  子组公平性：人口均等等差 / 机会均等等差
仅 Python 标准库，可离线审计。接 delivery-guard 门禁。
"""
from __future__ import annotations
import sys
import os
import io
import argparse
import csv
import math


def _rank_auc(y_true: list[int], y_score: list[float]) -> float:
    n = len(y_true)
    order = sorted(range(n), key=lambda i: y_score[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and y_score[order[j + 1]] == y_score[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    pos = sum(y_true)
    neg = n - pos
    if pos == 0 or neg == 0:
        return float("nan")
    s = sum(ranks[i] for i in range(n) if y_true[i] == 1)
    return (s - pos * (pos + 1) / 2.0) / (pos * neg)


def _brier(y_true: list[int], y_pred: list[float]) -> float:
    return sum((yp - yt) ** 2 for yp, yt in zip(y_pred, y_true)) / len(y_true)


def _threshold(y_pred: list[float], t: float = 0.5) -> list[int]:
    return [1 if p >= t else 0 for p in y_pred]


def metrics(y_true: list[int], y_pred: list[float]) -> dict:
    yhat = _threshold(y_pred)
    tp = sum(1 for a, b in zip(y_true, yhat) if a == 1 and b == 1)
    fp = sum(1 for a, b in zip(y_true, yhat) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y_true, yhat) if a == 1 and b == 0)
    tn = sum(1 for a, b in zip(y_true, yhat) if a == 0 and b == 0)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    acc = (tp + tn) / len(y_true)
    return {
        "auc": _rank_auc(y_true, y_pred),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "brier": _brier(y_true, y_pred),
    }


def fairness(y_true: list[int], y_pred: list[float], subgroup: list[str]) -> dict:
    groups = sorted(set(subgroup))
    stats = {}
    sel_rate = {}
    for g in groups:
        idx = [i for i in range(len(subgroup)) if subgroup[i] == g]
        yt = [y_true[i] for i in idx]
        yp = [y_pred[i] for i in idx]
        stats[g] = metrics(yt, yp)
        # 人口均等：各子组「正向预测率」（selection rate = P(ŷ=1)）之差
        yhat = _threshold(yp)
        sel_rate[g] = sum(yhat) / len(yhat) if yhat else 0.0
    # 机会均等：各子组 recall（TPR）之差
    rec = {g: stats[g]["recall"] for g in groups}
    return {
        "demographic_parity_gap": max(sel_rate.values()) - min(sel_rate.values()),
        "equal_opportunity_gap": max(rec.values()) - min(rec.values()),
        "by_group": stats,
    }


def _load(path: str, subgroup_col: str | None):
    """读 CSV。所有异常转成带上下文的 ValueError，供 main 友好报错
    （2026-10-12 压力测试修复：原实现对缺列/空文件/二进制/不存在文件均裸 traceback）。"""
    if not os.path.isfile(path):
        raise ValueError(f"数据文件不存在: {path}")
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            text = f.read()
    except UnicodeDecodeError:
        raise ValueError(f"{path} 不是 UTF-8 文本 CSV（疑似二进制或非 UTF-8 编码）")
    except OSError as e:
        raise ValueError(f"无法读取 {path}: {e}")
    r = csv.DictReader(io.StringIO(text, newline=""))
    fields = r.fieldnames or []
    missing = [c for c in ("y_true", "y_pred") if c not in fields]
    if missing:
        raise ValueError(f"{path} 缺列 {missing}（需表头 y_true,y_pred[,子组列]；实际表头 {fields}）")
    if subgroup_col and subgroup_col not in fields:
        raise ValueError(f"子组列 '{subgroup_col}' 不在表头 {fields}")
    yt, yp, sg = [], [], []
    for i, row in enumerate(r, 2):
        try:
            yt.append(int(float(row["y_true"])))
            yp.append(float(row["y_pred"]))
        except (TypeError, ValueError):
            raise ValueError(f"{path} 第 {i} 行非数值: "
                             f"y_true={row.get('y_true')!r} y_pred={row.get('y_pred')!r}")
        if yt[-1] not in (0, 1):
            raise ValueError(f"{path} 第 {i} 行 y_true={yt[-1]} 非二值（须 0/1，"
                             "否则 AUC/混淆矩阵口径不一致）")
        if not (0.0 <= yp[-1] <= 1.0):
            raise ValueError(f"{path} 第 {i} 行 y_pred={yp[-1]} 超出 [0,1]（AUC/Brier 前提）")
        if subgroup_col:
            sg.append(row.get(subgroup_col, ""))
    if not yt:
        raise ValueError(f"{path} 无数据行（空文件或仅表头）")
    return yt, yp, sg


def _demo() -> int:
    # 内置演示数据（100 样本二分类，含子组 A/B）
    yt = [1, 0, 1, 0, 1, 1, 0, 0, 1, 0] * 10
    yp = [0.9, 0.2, 0.8, 0.3, 0.6, 0.7, 0.1, 0.4, 0.85, 0.25] * 10
    sg = (["A"] * 5 + ["B"] * 5) * 10
    m = metrics(yt, yp)
    print("== 演示总体指标 ==")
    for k, v in m.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    fr = fairness(yt, yp, sg)
    print("\n== 演示公平性（子组 A/B）==")
    print(f"  人口均等等差: {fr['demographic_parity_gap']:.4f}")
    print(f"  机会均等等差: {fr['equal_opportunity_gap']:.4f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="医疗 AI 模型验证")
    ap.add_argument("--demo", action="store_true", help="跑内置演示数据（无需 --data）")
    ap.add_argument("--selftest", action="store_true", help="跑内置自测（断言 + 负对照 + xfail）")
    ap.add_argument("--data", help="CSV: 含 y_true,y_pred 列（演示可省略）")
    ap.add_argument("--subgroup", help="子组列名（可选，用于公平性）")
    args = ap.parse_args()
    if getattr(args, "selftest", False):
        return _selftest()
    if args.demo:
        return _demo()
    if not args.data:
        print("[err] 需 --data CSV 或 --demo 演示", file=sys.stderr)
        return 2
    try:
        yt, yp, sg = _load(args.data, args.subgroup)
    except ValueError as e:
        print(f"[error] {e}", file=sys.stderr)
        return 2
    m = metrics(yt, yp)
    print("== 总体指标 ==")
    for k, v in m.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    if args.subgroup and sg:
        fr = fairness(yt, yp, sg)
        print("\n== 公平性 ==")
        print(f"  人口均等等差: {fr['demographic_parity_gap']:.4f}")
        print(f"  机会均等等差: {fr['equal_opportunity_gap']:.4f}")
    return 0


# ==================== 内置自测（--selftest） ====================
# 由 conformance 补丁于 2026-10-14 生成（生成器 v2）。
# 特性：端到端 CLI 调用 + 内容断言 + 负对照 + xfail（已知缺陷显式标注，不掩盖）。
_SELFTEST_CASES = [(['--demo'], 0, [], [], None), ([], 2, [], [], None)]


def _selftest() -> int:
    import os as _os
    import subprocess as _sp
    _here = _os.path.dirname(_os.path.abspath(__file__))
    # 仓库根：脚本位于 <repo>/scripts/ 时取上一级；否则取脚本所在目录
    _root = _os.path.dirname(_here) if _os.path.basename(_here) == "scripts" else _here
    _script = _os.path.abspath(__file__)
    _fail, _xfail, _n = [], [], 0
    for _case in _SELFTEST_CASES:
        _args, _rc, _must, _mustnot = _case[0], _case[1], _case[2], _case[3]
        _reason = _case[4] if len(_case) > 4 else None
        _n += 1
        _p = _sp.run([sys.executable, _script] + _args, capture_output=True, cwd=_root)
        _out = _p.stdout.decode("utf-8", "replace")
        _bad = []
        if _p.returncode != _rc:
            _bad.append(f"rc={_p.returncode} 期望 {_rc}")
        for _m in _must:
            if _m not in _out:
                _bad.append(f"输出缺少 {_m!r}")
        for _m in _mustnot:
            if _m in _out:
                _bad.append(f"输出含禁止内容 {_m!r}")
        if _rc == 0 and not _out.strip():
            _bad.append("rc=0 但输出为空（假绿）")
        if _bad:
            if _reason:
                _xfail.append(f"xfail args={_args}：{'；'.join(_bad)} ← 已知缺陷：{_reason}")
            else:
                _fail.append(f"args={_args}：{'；'.join(_bad)}")
    # 通用负对照：未知参数必须非零
    _n += 1
    _p = _sp.run([sys.executable, _script, "--__nonexistent_flag__"], capture_output=True, cwd=_root)
    if _p.returncode == 0:
        _fail.append("负对照失败：未知参数竟返回 0")
    for _x in _xfail:
        print("[selftest][xfail] " + _x)
    if _fail:
        print(f"[selftest] FAILED {len(_fail)}/{_n}")
        for _f in _fail:
            print("  - " + _f)
        return 1
    print(f"[selftest] PASS {_n - len(_xfail)}/{_n}（含负对照；xfail {len(_xfail)} 项为已知缺陷）")
    return 0


if __name__ == "__main__":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    raise SystemExit(main())
