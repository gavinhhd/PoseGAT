import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from efficient_kan import KANLinear


# ============ 1) 常量 & 标签映射 ============
CSV_PATH    = Path("mvs_augmented_full.csv")
save_path   = "./ckpt_kan_fixed"
os.makedirs(save_path, exist_ok=True)

SUBJECT_COL = 0                                  # 受试者列（与你原脚本保持一致）
UPPER_LBLS  = ["TUP","TLB","TLF","TLR","TLL"]    # 5
LOWER_LBLS  = ["LAP","LWA","LCS","LCR","LCL","LLR","LLL"]  # 7
UPPER_MAP   = {k:i for i,k in enumerate(UPPER_LBLS)}
LOWER_MAP   = {k:i for i,k in enumerate(LOWER_LBLS)}

# 只保留 11 个关节 (nose + shoulder/hip/knee/ankle/heel) → 33 维
JOINTS = [
    "nose",
    "left_shoulder","right_shoulder",
    "left_hip","right_hip",
    "left_knee","right_knee",
    "left_ankle","right_ankle",
    "left_heel","right_heel"
]
FEATURE_COLS = [f"{j}_{c}" for j in JOINTS for c in ("x","y","z")]

def encode(series, mapping):
    return series.map(mapping).to_numpy(np.int64)

def select_X33(df):
    return df[FEATURE_COLS].to_numpy(np.float32)


# ============ 2) 受试者分组外层划分（无 kfold） ============
def build_outer_splits(subjects, mode="group_9v4", repeats=8, seed=2025, train_n=9, test_n=4):
    rng = np.random.RandomState(seed)
    subs = list(subjects)
    splits = []
    if mode == "loso":
        for sid in subs:
            splits.append(dict(name=f"test[{sid}]", test=[sid], train=[s for s in subs if s != sid]))
    elif mode == "group_9v4":
        assert len(subs) >= train_n + test_n, "受试者数量不足以进行 group_9v4 划分"
        for r in range(repeats):
            perm = subs.copy(); rng.shuffle(perm)
            test = perm[:test_n]
            train = perm[test_n:test_n+train_n]
            splits.append(dict(name=f"9v4-r{r+1}", test=test, train=train))
    elif mode == "group_fixed":
        assert len(subs) >= train_n + test_n, "受试者数量不足以进行 group_fixed 划分"
        for r in range(repeats):
            perm = subs.copy(); rng.shuffle(perm)
            test = perm[:test_n]
            train = perm[test_n:test_n+train_n]
            splits.append(dict(name=f"fixed-{train_n}v{test_n}-r{r+1}", test=test, train=train))
    else:
        raise ValueError(f"Unknown outer_mode: {mode}")
    return splits


# ============ 2.1) 新增：解析固定 ID 列表 ============
def parse_id_list(id_str, subj_dtype):
    """把 '0,3,5,8' 解析成与受试者列 dtype 对齐的列表"""
    if id_str is None or id_str.strip() == "":
        return None
    parts = [p.strip() for p in id_str.split(",") if p.strip() != ""]
    if pd.api.types.is_integer_dtype(subj_dtype):
        return [int(p) for p in parts]
    elif pd.api.types.is_float_dtype(subj_dtype):
        return [float(p) for p in parts]
    else:
        return parts  # 字符串/类别型


# ============ 3) 数据集 ============
class PostureDS(Dataset):
    def __init__(self, X, yu, yl):
        # 确保类型正确
        self.X  = torch.tensor(X, dtype=torch.float32)
        self.yu = torch.tensor(yu, dtype=torch.long)
        self.yl = torch.tensor(yl, dtype=torch.long)
    def __len__(self):          return len(self.X)
    def __getitem__(self, i):   return self.X[i], self.yu[i], self.yl[i]


# ============ 4) KAN 多任务网络（保持你的结构不变） ============
class MultiTaskKAN(nn.Module):
    def __init__(self, in_dim=33, hidden=12, n_layers=1,
                 n_u=5, n_l=7, grid=3, spline_order=3):
        super().__init__()
        layers = []
        dim = in_dim
        for _ in range(n_layers):
            layers.append(KANLinear(dim, hidden, grid_size=grid, spline_order=spline_order))
            dim = hidden
        self.backbone = nn.Sequential(*layers)
        self.head_u   = nn.Linear(hidden, n_u)
        self.head_l   = nn.Linear(hidden, n_l)

    def forward(self, x):
        h = self.backbone(x)
        return self.head_u(h), self.head_l(h)


# ============ 5) 训练 / 评估（修复 train 判断 & 测试 no_grad） ============
def run_epoch(net, loader, ce, opt=None, dev="cpu"):
    # 关键修复：只有当 opt 真的像优化器时才视为训练分支
    train = (opt is not None) and hasattr(opt, "zero_grad")
    net.train() if train else net.eval()

    losses, preds_u, preds_l, gts_u, gts_l = [], [], [], [], []

    for X, yu, yl in loader:
        X, yu, yl = X.to(dev), yu.to(dev), yl.to(dev)
        if train:
            ou, ol = net(X)
            loss = ce(ou, yu) + ce(ol, yl)
            opt.zero_grad(); loss.backward(); opt.step()
        else:
            with torch.no_grad():
                ou, ol = net(X)
                loss = ce(ou, yu) + ce(ol, yl)
        losses.append(loss.item())
        preds_u += ou.argmax(1).cpu().tolist()
        preds_l += ol.argmax(1).cpu().tolist()
        gts_u   += yu.cpu().tolist()
        gts_l   += yl.cpu().tolist()

    return float(np.mean(losses)), accuracy_score(gts_u, preds_u), accuracy_score(gts_l, preds_l)


# ============ 6) 主流程：支持 fixed list / group_9v4 / group_fixed / loso（默认 9v4） ============
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(CSV_PATH))
    ap.add_argument("--outer_mode", choices=["group_9v4","group_fixed","loso"], default="group_9v4")
    ap.add_argument("--outer_repeats", type=int, default=8)
    ap.add_argument("--train_n", type=int, default=9)
    ap.add_argument("--test_n", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=0.006)
    ap.add_argument("--weight_decay", type=float, default=3.321e-6)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--save_dir", default=save_path)
    # 新增：固定测试/训练 ID 列表
    ap.add_argument("--fixed_test_ids", type=str, default="1,2,3,4")
    ap.add_argument("--fixed_train_ids", type=str, default="")
    args = ap.parse_args()

    # reproducibility
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    # 读取 CSV
    df = pd.read_csv(args.csv)

    # 标签编码（与你脚本一致：第 2 / 第 3 列）
    y_u = encode(df.iloc[:, 1], UPPER_MAP)
    y_l = encode(df.iloc[:, 2], LOWER_MAP)

    # 特征 33 维
    X = select_X33(df)

    # 设备
    dev = args.device if args.device in ("cpu", "cuda") else "cpu"

    # 受试者列
    subj_col = df.iloc[:, SUBJECT_COL]
    subjects = subj_col.unique()

    # ===== 固定列表：若提供 --fixed_test_ids，则忽略 outer_mode/outer_repeats，生成 1 折 =====
    fixed_test  = parse_id_list(args.fixed_test_ids, subj_col.dtype)
    fixed_train = parse_id_list(args.fixed_train_ids, subj_col.dtype)

    if fixed_test is not None and len(fixed_test) > 0:
        # 校验 test 是否在数据中
        missing = [sid for sid in fixed_test if sid not in subjects]
        if missing:
            raise ValueError(f"以下测试ID不在数据里: {missing}")
        if fixed_train is None or len(fixed_train) == 0:
            # 其余全部做训练
            fixed_train = [sid for sid in subjects if sid not in set(fixed_test)]
        else:
            miss_tr = [sid for sid in fixed_train if sid not in subjects]
            if miss_tr:
                raise ValueError(f"以下训练ID不在数据里: {miss_tr}")
            # 不允许 train/test 重叠
            if set(fixed_test) & set(fixed_train):
                overlap = list(set(fixed_test) & set(fixed_train))
                raise ValueError(f"训练/测试 ID 有重叠: {overlap}")
        splits = [dict(name="fixedlist", test=fixed_test, train=fixed_train)]
        print(f"[FixedList] test={fixed_test} | train={fixed_train}")
    else:
        # 保持原有随机外层划分
        splits = build_outer_splits(
            subjects, mode=args.outer_mode, repeats=args.outer_repeats,
            seed=args.seed, train_n=args.train_n, test_n=args.test_n
        )

    uppers, lowers, rows, fold_names = [], [], [], []

    for sp in splits:
        test_subjects  = sp["test"]
        train_subjects = sp["train"]

        test_mask  = subj_col.isin(test_subjects).to_numpy()
        train_mask = subj_col.isin(train_subjects).to_numpy()

        # 每折：仅用训练折拟合 StandardScaler
        scaler = StandardScaler().fit(X[train_mask])
        Xtr = scaler.transform(X[train_mask])
        Xte = scaler.transform(X[test_mask])

        tr_loader = DataLoader(PostureDS(Xtr, y_u[train_mask], y_l[train_mask]),
                               batch_size=args.batch_size, shuffle=True)
        te_loader = DataLoader(PostureDS(Xte, y_u[test_mask],  y_l[test_mask]),
                               batch_size=args.batch_size, shuffle=False)

        net = MultiTaskKAN().to(dev)
        ce  = nn.CrossEntropyLoss()
        opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.856)

        # 训练
        for _ in range(args.epochs):
            run_epoch(net, tr_loader, ce, opt=opt, dev=dev)
            scheduler.step()

        # 测试（明确 opt=None, dev=dev）
        _, acc_u, acc_l = run_epoch(net, te_loader, ce, opt=None, dev=dev)

        uppers.append(acc_u); lowers.append(acc_l); fold_names.append(sp["name"])
        rows.append(dict(
            fold=sp["name"],
            test_subjects=",".join(map(str, test_subjects)),
            train_subjects=",".join(map(str, train_subjects)),
            upper_acc=acc_u, lower_acc=acc_l,
            n_train=int(train_mask.sum()), n_test=int(test_mask.sum())
        ))
        print(f"[{sp['name']}] Upper {acc_u:.3%}  Lower {acc_l:.3%}  (train={int(train_mask.sum())}, test={int(test_mask.sum())})")

    # ==== 保存 CSV / JSON ====
    dfm = pd.DataFrame(rows)
    prefix = splits[0]['name'] if len(splits)==1 and splits[0]['name']=="fixedlist" else args.outer_mode
    csv_path = os.path.join(args.save_dir, f"outer_{prefix}_per_fold_kan.csv")
    dfm.to_csv(csv_path, index=False)
    metrics = dict(
        folds=fold_names, upper_acc=uppers, lower_acc=lowers,
        upper_avg=float(np.nanmean(uppers) if uppers else float('nan')),
        lower_avg=float(np.nanmean(lowers) if lowers else float('nan'))
    )
    json_path = os.path.join(args.save_dir, f"outer_{prefix}_metrics_kan.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # ==== 绘图：单图合并柱状；柱内显示百分比 ====
    ids = np.arange(len(fold_names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(11, 6))

    ub = ax.bar(ids - w / 2, uppers, w, label="Upper")
    lb = ax.bar(ids + w / 2, lowers, w, label="Lower")

    for b in list(ub) + list(lb):
        ax.text(
            b.get_x() + b.get_width() / 2,
            max(b.get_height() / 2, 0.01),
            f"{b.get_height() * 100:.2f}%",
            ha="center", va="center", fontsize=10, rotation=90
        )

    ax.set_xticks(ids)
    ax.set_xticklabels(fold_names, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Fold")
    ax.set_ylabel("Accuracy")
    title_mode = {"group_9v4":"9/13 Train vs 4/13 Test","group_fixed":"Fixed Train/Test","loso":"LOSO","fixedlist":"Fixed Test IDs"}.get(prefix, prefix)
    ax.set_title(f"KAN Results with Subject-Group Splits ({title_mode})", fontsize=14)
    ax.legend(loc="lower left", fontsize=10)
    plt.tight_layout()
    png_name = f"outer_{prefix}_bars_kan.png"
    plt.savefig(os.path.join(args.save_dir, png_name), dpi=150)
    plt.close()

    print("\n=== 外层平均准确率 ===")
    print(f"Upper: {np.mean(uppers):.3%}  |  Lower: {np.mean(lowers):.3%}")
