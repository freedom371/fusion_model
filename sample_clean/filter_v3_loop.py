# -*- coding: utf-8 -*-
"""新口径 (权重选择依据 = train AUC 最大) 下的 train 集筛选。

与旧循环的区别: 不再关心"固定 checkpoint + 选子集", 而是**每个候选子集真重训**,
指标一律取"train AUC 最大的那个 epoch"的 train AUC/ACC —— 这正是 train.py 现在的
权重选择依据, 也是训练日志里打印的口径。

指标来源: 直接读 train.py 写出的 history_{ts}_seed{seed}.csv, 取 epoch>6 中
train_auc 最大的那一行。不重算, 保证与 train.py 的选择完全一致。

用法:
  python filter_v3_loop.py --bracket        # 删最易正样本 k=77/110/140
  python filter_v3_loop.py --k 110 200      # 自定义
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path("/home/guozheng/claude_dir/wsi_research")
sys.path.insert(0, str(ROOT / "fusion_model"))
sys.path.insert(0, str(ROOT / "sample_clean"))

import dataset as fu_ds  # noqa: E402
from label_l_retrain_loop import POOL, LABELS, write_label  # noqa: E402

OUT = ROOT / "sample_clean/label_l_out"
V3_DIR = OUT / "filter_v3"
V3_DIR.mkdir(parents=True, exist_ok=True)
CSV = V3_DIR / "results.csv"
EXP_NAME = "cmt_cpg_v3"
EXP_DIR = ROOT / "fusion_model/output" / EXP_NAME
REF_CKPT = ROOT / "fusion_model/output/cmt_cpg/checkpoints/best_model_seed1013.pt"
LO, HI = 0.30, 0.40
TARGET = 0.80
EPOCHS_OVERRIDE = None      # --epochs N: 临时覆盖训练轮数(诊断用, 不改 train.py)


def ref_scores():
    """用当前最佳权重给池内 520 例打分 (参考排序, 全程固定)。

    注意: 打分必须在任何重训之前完成并落盘缓存 —— 重训会覆盖 REF_CKPT。
    """
    cache = V3_DIR / "ref_scores.csv"
    if cache.exists():
        d = pd.read_csv(cache, dtype={"base_id": str})
        return dict(zip(d.base_id, d.score))

    ids443 = pd.read_csv(ROOT / "data/cvs/label_l_filtered.csv", dtype={"ID": str})
    ids443 = ids443[ids443.group == "train"]["ID"].str.replace(
        r"\.nii\.gz$", "", regex=True).astype(str).tolist()
    from label_l_retrain_loop import predict_pool
    preds = predict_pool(REF_CKPT, ids443)          # 与 POOL 顺序对齐
    pd.DataFrame({"base_id": POOL, "score": preds}).to_csv(cache, index=False)
    return dict(zip(POOL, preds))


def build_set(scores, k_easy=0, k_hard=0, k_neg_easy=0):
    """从 520 池里删样本。k_easy/k_hard: 删分数最高/最低的正样本;
    k_neg_easy: 删分数最低的负样本(会抬高占比, 需配合多删正样本)。"""
    pos = sorted([i for i in POOL if LABELS[i] == 1], key=lambda i: scores[i])
    neg = sorted([i for i in POOL if LABELS[i] == 0], key=lambda i: scores[i])
    drop = set()
    if k_easy:
        drop |= set(pos[-k_easy:])
    if k_hard:
        drop |= set(pos[:k_hard])
    if k_neg_easy:
        drop |= set(neg[:k_neg_easy])
    return [i for i in POOL if i not in drop]


def read_history(exp_dir):
    """读 train.py 写的最新 history csv, 返回 train_auc 最大的 epoch 那行。"""
    files = sorted(exp_dir.glob("history_*_seed1013.csv"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise RuntimeError(f"未找到 history csv: {exp_dir}")
    h = pd.read_csv(files[-1])
    h = h[h.epoch > 6]
    best = h.loc[h.train_auc.idxmax()]
    return best, files[-1].name


def run_candidate(tag, keep):
    y = np.array([LABELS[i] for i in keep]).astype(int)
    ratio = float(y.mean())

    import train as train_mod
    train_mod.conf["exp_name"] = EXP_NAME
    if EPOCHS_OVERRIDE:
        train_mod.conf["num_epochs"] = EPOCHS_OVERRIDE
    wd = float(train_mod.conf.get("weight_decay", -1.0))
    if wd > 0:                       # tag 带 wd: 换 wd 重扫时旧行不被同名覆盖
        tag = f"{tag}_wd{wd:.0e}"

    print(f"\n{'='*74}\n=== {tag}: n={len(keep)} pos={y.sum()} ratio={ratio:.4f} ===\n{'='*74}")
    label_file = write_label(keep, V3_DIR / f"label_{tag}.csv")
    fu_ds.LABEL_CSV = label_file
    fu_ds._TRAIN_STATS = None
    train_mod._train_impl(train_mod.conf)

    best, hname = read_history(EXP_DIR)
    shutil.copy(EXP_DIR / "checkpoints/best_model_seed1013.pt", V3_DIR / f"best_{tag}.pt")

    # 干净口径复核 + 最优阈值 ACC (偏置校准后 ACC@0.5 即等于它, 是阈值无关的判别力度量)
    from label_l_probe import best_acc, predict_split
    p_ids, p_pred = predict_split(V3_DIR / f"best_{tag}.pt", keep, "train")
    p = dict(zip(p_ids, p_pred))
    ss = np.array([p[i] for i in keep])
    from label_l_retrain_loop import auc_acc as _auc_acc
    ca, cc = _auc_acc(y, ss)
    ba, bt = best_acc(ss, y)
    print(f">>> [{tag}] 选中 epoch={int(best.epoch)} | 日志口径 train AUC={best.train_auc:.4f} "
          f"ACC={best.train_acc:.4f} | val {best.val_auc:.4f}/{best.val_acc:.4f} "
          f"| test {best.test_auc:.4f}/{best.test_acc:.4f}")
    print(f"    干净口径 train AUC={ca:.4f} ACC@0.5={cc:.4f} | 最优阈值ACC={ba:.4f} "
          f"(阈值{bt:.4f})  ←校准后ACC即此值")
    Path(V3_DIR / f"keep_{tag}.txt").write_text("\n".join(keep), encoding="utf-8")

    rec = {"tag": tag, "wd": wd, "n": len(keep), "pos": int(y.sum()), "ratio": ratio,
           "epoch": int(best.epoch), "train_auc": float(best.train_auc),
           "train_acc": float(best.train_acc), "val_auc": float(best.val_auc),
           "val_acc": float(best.val_acc), "test_auc": float(best.test_auc),
           "test_acc": float(best.test_acc),
           "clean_auc": float(ca), "clean_acc": float(cc),
           "best_acc": float(ba), "best_thr": float(bt),
           "cost": abs(best.train_auc - TARGET) + abs(ba - TARGET),
           "ok": bool(abs(best.train_auc - TARGET) <= 0.02
                      and abs(ba - TARGET) <= 0.02 and LO <= ratio <= HI)}
    hist = pd.read_csv(CSV) if CSV.exists() else pd.DataFrame(columns=list(rec))
    if "tag" not in hist.columns:                     # 空表(首次运行)
        hist = pd.DataFrame(columns=list(rec))
    hist = pd.concat([hist[hist.tag != tag], pd.DataFrame([rec])], ignore_index=True)
    hist.to_csv(CSV, index=False)
    return rec


if __name__ == "__main__":
    scores = ref_scores()
    print(f"[参考] {REF_CKPT.name} 已给 {len(scores)} 例池内样本打分")

    if "--epochs" in sys.argv:
        EPOCHS_OVERRIDE = int(sys.argv[sys.argv.index("--epochs") + 1])
        print(f"[诊断] 训练轮数临时覆盖为 {EPOCHS_OVERRIDE}")

    # ── 模式 B: 从 520 池随机删 K 个正样本 (删除量最小路线) ──
    if "--rand520" in sys.argv:
        K = int(sys.argv[sys.argv.index("--rand520") + 1])
        SEED = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 1013
        pos_all = sorted([i for i in POOL if LABELS[i] == 1])
        rng = np.random.RandomState(SEED)
        drop = set(rng.choice(pos_all, size=K, replace=False).tolist())
        keep = [i for i in POOL if i not in drop]
        y = np.array([LABELS[i] for i in keep]).astype(int)
        if not (LO <= y.mean() <= HI):
            raise SystemExit(f"占比 {y.mean():.4f} 超出 [{LO},{HI}]，K={K} 不可用")
        import os
        os.chdir(ROOT / "sample_clean")
        run_candidate(f"rand520_{K}_s{SEED}" + (f"_ep{EPOCHS_OVERRIDE}" if EPOCHS_OVERRIDE else ""), keep)
        print("\n汇总:")
        print(pd.read_csv(CSV).to_string(index=False))
        raise SystemExit(0)

    # ── 模式 A: 从 443 集随机删 K 个正样本 (复刻旧 rand380) ──
    if "--rand443" in sys.argv:
        K = int(sys.argv[sys.argv.index("--rand443") + 1])
        d = pd.read_csv(ROOT / "data/cvs/label_l_filtered.csv", dtype={"ID": str})
        ids443 = d[d.group == "train"]["ID"].str.replace(
            r"\.nii\.gz$", "", regex=True).astype(str).tolist()
        pos443 = sorted([i for i in ids443 if LABELS[i] == 1])
        rng = np.random.RandomState(1013)
        drop = set(rng.choice(pos443, size=K, replace=False).tolist())
        keep = [i for i in ids443 if i not in drop]
        import os
        os.chdir(ROOT / "sample_clean")
        run_candidate(f"rand443_{K}" + (f"_ep{EPOCHS_OVERRIDE}" if EPOCHS_OVERRIDE else ""), keep)
        print("\n汇总:")
        print(pd.read_csv(CSV).to_string(index=False))
        raise SystemExit(0)

    # ── 模式 C: 用外部 keep 列表重训 (pick_harmful.py 等产出的候选) ──
    if "--keep" in sys.argv:
        p = Path(sys.argv[sys.argv.index("--keep") + 1])
        keep = [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        if not keep:
            raise SystemExit(f"keep 列表为空: {p}")
        tag = p.stem.replace("keep_", "")
        import os
        os.chdir(ROOT / "sample_clean")
        run_candidate(tag, keep)
        print("\n汇总:")
        print(pd.read_csv(CSV).to_string(index=False))
        raise SystemExit(0)

    ks = []
    if "--bracket" in sys.argv:
        ks = [(77, 0, 0), (110, 0, 0), (140, 0, 0)]
    else:
        k = int(sys.argv[sys.argv.index("--k") + 1])
        ks = [(k, 0, 0)]
    # 已跑过的 tag 跳过 (支持中断后续跑)
    if "--skip-done" in sys.argv and CSV.exists():
        done = set(pd.read_csv(CSV).tag)
        ks = [(a, b, c) for a, b, c in ks if f"e{a}_h{b}_n{c}" not in done]

    for k_easy, k_hard, k_neg in ks:
        tag = f"e{k_easy}_h{k_hard}_n{k_neg}"
        run_candidate(tag, build_set(scores, k_easy, k_hard, k_neg))

    print("\n汇总:")
    print(pd.read_csv(CSV).to_string(index=False))
    print(f"\n→ {CSV}")
