#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gold set setup helper：在模块仓根写 _gold_sample.csv 供 ml-12 真实数据路径测试。"""
import csv, os
OUT = os.path.join(os.getcwd(), "_gold_sample.csv")
rows = [
    ("y_true", "y_pred", "group"),
    (1, 0.9, "A"), (0, 0.2, "A"), (1, 0.8, "A"), (0, 0.1, "A"), (1, 0.7, "A"),
    (1, 0.6, "B"), (0, 0.4, "B"), (1, 0.85, "B"), (0, 0.3, "B"), (0, 0.15, "B"),
]
with open(OUT, "w", encoding="utf-8", newline="") as f:
    csv.writer(f).writerows(rows)
print("wrote", OUT)
