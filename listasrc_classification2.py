# listasrc_classification2.py
# -----------------------------------------------------------
# 音色分类 (Mel 特征 → PCA → Coupled‑LISTA‑SRC)
# -----------------------------------------------------------
import torch
import numpy as np
from dataset import AudioInstrumentDataset
from model import SRCMelFeatureExtractor
from constants import SAMPLE_RATE, DEFAULT_DEVICE
from listasrc2 import SRCClassifierLISTA            # ← 使用刚更新的模块
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ---------- 基本设置 ---------- #
sequence_length = int(SAMPLE_RATE * 0.5)            # 0.5 秒截断
train_ds = AudioInstrumentDataset("train_metadata.csv", sequence_length)
test_ds  = AudioInstrumentDataset("test_metadata.csv",  sequence_length)

feature_extractor = SRCMelFeatureExtractor(sample_rate=SAMPLE_RATE).to(DEFAULT_DEVICE)

# ---------- 提取特征 ---------- #

def extract_features(ds):
    feats, lbls = [], []
    for waveform, onehot in ds:
        vec = feature_extractor(waveform.unsqueeze(0))      # (1, feat_dim)
        feats.append(vec.squeeze(0).cpu().numpy())
        lbls.append(onehot.argmax().item())
    return np.asarray(feats, dtype=np.float32).T, np.asarray(lbls)

train_feats, train_lbls = extract_features(train_ds)        # (feat_dim, N_train)
test_feats , test_lbls  = extract_features(test_ds)         # (feat_dim, N_test)
print("The shape of train_feats are:{}".format(train_feats.shape))
print("The shape of train_lbls are:{}".format(train_lbls.shape))
# demo：只取前 20 个测试样本
test_feats, test_lbls   = test_feats[:, :20], test_lbls[:20]

# ---------- PCA 降维 ---------- #

def pca_projection(X, keep_energy: float = 0.999):
    Xn = X / (np.linalg.norm(X, axis=0, keepdims=True) + 1e-8)
    cov = Xn @ Xn.T
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
    energy = np.cumsum(eigvals) / eigvals.sum()
    k = np.argmax(energy >= keep_energy) + 1
    V = eigvecs[:, :k]                   # (feat_dim, k)
    return V.T @ Xn, V                   # (k, num), (feat_dim, k)

train_pca, V = pca_projection(train_feats)
# 测试集同一投影
Xn_test = test_feats / (np.linalg.norm(test_feats, axis=0, keepdims=True) + 1e-8)
test_pca = V.T @ Xn_test

# ---------- 转成 Torch ---------- #
train_torch = torch.from_numpy(train_pca).to(DEFAULT_DEVICE)   # (k, N)
test_torch  = torch.from_numpy(test_pca).to(DEFAULT_DEVICE)    # (k, T)

# ---------- LISTA‑SRC 训练 ---------- #
listasrc = SRCClassifierLISTA(
    depth=12,
    lam=0.08,
    learn_theta=True,     # 学 θ_k
    learn_B=True,         # 学 B=W1
    finetune_epochs=30,
    lr_theta=1e-3,
    lr_B=5e-4,
    device=DEFAULT_DEVICE,
)
listasrc.fit(train_torch, train_lbls)

# ---------- 预测 & 评估 ---------- #
preds = listasrc.predict(test_torch)
acc = (preds == test_lbls).mean()
print(f"Test Accuracy via LISTA‑SRC + PCA = {acc:.2%}")

cm = confusion_matrix(test_lbls, preds)
disp = ConfusionMatrixDisplay(cm, display_labels=[
    "guitar","flute","violin","clarinet","trumpet","cello","saxophone"])
fig, ax = plt.subplots(figsize=(8,6))
disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix – LISTA‑SRC")
plt.tight_layout()
plt.savefig("confusion_matrix_lista.png")
plt.show()
