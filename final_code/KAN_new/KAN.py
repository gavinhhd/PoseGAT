import json
import os
import random
import matplotlib.pyplot as plt
import pandas as pd, numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader
import torch, torch.nn as nn
from efficient_kan import KANLinear
import matplotlib
matplotlib.use('Agg')

# ============ 1. 常量 & 标签映射 ============
CSV_PATH    = Path("mvs_augmented_full.csv")
save_path="./ckpt_kan"
png_path = os.path.join(save_path, "loso_upper_lower_bars_kan.png")

SUBJECT_COL = 0                                  # 受试者列
UPPER_LBLS  = ["TUP","TLB","TLF","TLR","TLL"]    # 5
LOWER_LBLS  = ["LAP","LWA","LCS","LCR","LCL","LLR","LLL"]  # 7
UPPER_MAP   = {k:i for i,k in enumerate(UPPER_LBLS)}
LOWER_MAP   = {k:i for i,k in enumerate(LOWER_LBLS)}

JOINTS = [
    "nose",
    "left_shoulder","right_shoulder",
    "left_hip","right_hip",
    "left_knee","right_knee",
    "left_ankle","right_ankle",
    "left_heel","right_heel"
]
FEATURE_COLS = [f"{j}_{c}" for j in JOINTS for c in ("x","y","z")]

def encode(s, m): return s.map(m).to_numpy(np.int64)
def X33(df):       return df[FEATURE_COLS].to_numpy(np.float32)

# ============ 2. 数据集 ============
class PostureDS(Dataset):
    def __init__(self, X, yu, yl):
        self.X, self.yu, self.yl = map(torch.tensor, (X, yu, yl))
    def __len__(self):          return len(self.X)
    def __getitem__(self, i):   return self.X[i], self.yu[i], self.yl[i]

# ============ 3. KAN 多任务网络 ============
class MultiTaskKAN(nn.Module):
    def __init__(self, in_dim=33, hidden=12, n_layers=1,
                 n_u=5, n_l=7, grid=3, spline_order=3):
        super().__init__()
        layers = []
        dim = in_dim
        for _ in range(n_layers):
            layers.append(
                KANLinear(dim, hidden,grid_size=grid, spline_order=spline_order)        # 权重由 spline 控制，可不再要 bias
            )
            dim = hidden
        self.backbone = nn.Sequential(*layers)
        self.head_u   = nn.Linear(hidden, n_u)
        self.head_l   = nn.Linear(hidden, n_l)

    def forward(self, x):
        h = self.backbone(x)
        return self.head_u(h), self.head_l(h)

# ============ 4. 训练/评估 ============
def run_epoch(net, loader, ce, opt=None):
    train = opt is not None
    net.train() if train else net.eval()
    losses, preds_u,preds_l, gts_u,gts_l = [], [], [], [], []
    for X,yu,yl in loader:
        X,yu,yl = X.to(dev), yu.to(dev), yl.to(dev)
        with torch.set_grad_enabled(train):
            ou, ol = net(X)
            loss = ce(ou,yu)+ce(ol,yl)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
        preds_u += ou.argmax(1).cpu().tolist()
        preds_l += ol.argmax(1).cpu().tolist()
        gts_u   += yu.cpu().tolist()
        gts_l   += yl.cpu().tolist()
    return np.mean(losses), accuracy_score(gts_u,preds_u), accuracy_score(gts_l,preds_l)

# ============ 5. LOSO 主流程 ============
if __name__ == "__main__":

    seed = 1
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    df = pd.read_csv(CSV_PATH)
    y_u = encode(df.iloc[:,1], UPPER_MAP)
    y_l = encode(df.iloc[:,2], LOWER_MAP)
    X99 = X33(df)

    #dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = 'cpu'
    sub_ids = df.iloc[:,SUBJECT_COL].unique()
    uppers,lowers = [], []
    rows = []
    for sid in sub_ids:
        test_mask = df.iloc[:,SUBJECT_COL]==sid
        scaler = StandardScaler().fit(X99[~test_mask])
        Xtr = scaler.transform(X99[~test_mask]); Xte = scaler.transform(X99[test_mask])

        tr_loader = DataLoader(PostureDS(Xtr,y_u[~test_mask],y_l[~test_mask]),
                               batch_size=256, shuffle=True)
        te_loader = DataLoader(PostureDS(Xte,y_u[test_mask],y_l[test_mask]),
                               batch_size=256, shuffle=False)

        net = MultiTaskKAN().to(dev)
        ce  = nn.CrossEntropyLoss()
        opt = torch.optim.AdamW(net.parameters(), lr=0.006, weight_decay=3.321e-6)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.856)

        for _ in range(20):
            run_epoch(net,tr_loader,ce,opt); scheduler.step()
        _, acc_u, acc_l = run_epoch(net,te_loader,ce)
        uppers.append(acc_u); lowers.append(acc_l)
        rows.append(dict(subject=sid, upper_acc=acc_u, lower_acc=acc_l, n_train=len(Xtr), n_test=len(Xte)))
        print(f"Subject {sid:>2}:  Upper {acc_u:.3%}  Lower {acc_l:.3%}")

    #数据打印
    dfm = pd.DataFrame(rows)
    csv_path = os.path.join(save_path, "loso_per_subject_kan.csv"); dfm.to_csv(csv_path, index=False)
    metrics = dict(subjects=sub_ids.tolist(), upper_acc=uppers, lower_acc=lowers,
                   upper_avg=float(np.nanmean(uppers) if uppers else float('nan')),
                   lower_avg=float(np.nanmean(lowers) if lowers else float('nan')))
    json_path = os.path.join(save_path, "loso_metrics_kan.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)



    # ============ 6. 绘图 ============
    ids = np.arange(len(sub_ids))
    w = 0.35  # 条形宽度

    fig, ax = plt.subplots(figsize=(11, 6))

    # 绘制条形
    ub = ax.bar(ids - w / 2, uppers, w, label="Upper", color="#74b9ff")
    lb = ax.bar(ids + w / 2, lowers, w, label="Lower", color="#ff9f7f")

    # 在柱子内部添加百分比
    for b in ub + lb:
        ax.text(
            b.get_x() + b.get_width() / 2,  # x 位置：柱子中点
            b.get_height() / 2,  # y 位置：柱子中间
            f"{b.get_height() * 100:.2f}%",  # 百分比
            ha="center", va="center",  # 文字居中对齐
            color="black", fontsize=10,
            rotation = 90  # ⬅ 文字旋转90度
        )

    # 设置 x 轴和标题
    ax.set_xticks(ids)
    ax.set_xticklabels(sub_ids, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Participant ID")
    ax.set_ylabel("Accuracy")
    ax.set_title("KAN Results with Test Set for Each Participant (LOSO)", fontsize=14)

    # 图例放左下角
    ax.legend(loc="lower left", fontsize=10)
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()
    #plt.show()

    print("\n=== LOSO 平均准确率 ===")
    print(f"Upper: {np.mean(uppers):.3%}  |  Lower: {np.mean(lowers):.3%}")