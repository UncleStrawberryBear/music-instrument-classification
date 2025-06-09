# listasrc_classification.py
# -----------------------------------------------------------
# 音色分类 (特征提取 + PCA + LISTA-SRC)
# -----------------------------------------------------------
import torch
import numpy as np
from dataset import AudioInstrumentDataset
from model import SRCMelFeatureExtractor
from constants import SAMPLE_RATE, DEFAULT_DEVICE
from listasrc import SRCClassifierLISTA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ---------- 基本设置 ---------- #
sequence_length = int(SAMPLE_RATE * 0.5)   # 0.5 秒截断
train_ds = AudioInstrumentDataset("train_metadata.csv", sequence_length)
test_ds  = AudioInstrumentDataset("test_metadata.csv",  sequence_length)

feature_extractor = SRCMelFeatureExtractor(sample_rate=SAMPLE_RATE).to(DEFAULT_DEVICE)

# ---------- 提取特征 ---------- #
def extract_features(ds):
    feats, lbls = [], []
    for waveform, onehot in ds:
        vec = feature_extractor(waveform.unsqueeze(0))
        feats.append(vec.squeeze(0).cpu().numpy())
        lbls.append(onehot.argmax().item())
    return np.array(feats, dtype=np.float32).T, np.array(lbls)

train_feats, train_lbls = extract_features(train_ds)    # (feat_dim, num_train)
test_feats , test_lbls  = extract_features(test_ds)      # (feat_dim, num_test)
# debug：只取前 20 个样本
test_feats, test_lbls   = test_feats[:, :20], test_lbls[:20]

# ---------- PCA 降维 (保留 99.9% 能量) ---------- #
def pca_projection(X, energy_keep=0.999):
    # L2 规范列
    Xn = X / (np.linalg.norm(X, axis=0, keepdims=True) + 1e-8)
    cov = Xn @ Xn.T
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
    energy = np.cumsum(eigvals) / eigvals.sum()
    k = np.argmax(energy >= energy_keep) + 1
    V = eigvecs[:, :k]          # (feat_dim, k)
    return V.T @ Xn, V          # (k, num), (k, feat_dim)

train_pca, V = pca_projection(train_feats)
#test_pca  = V @ (test_feats / (np.linalg.norm(test_feats, axis=0, keepdims=True)+1e-8))
test_pca = V.T @ (test_feats / (np.linalg.norm(test_feats, axis=0, keepdims=True) + 1e-8))

# ---------- 转成 Torch ---------- #
train_torch = torch.from_numpy(train_pca).to(DEFAULT_DEVICE)   # (k, N)
test_torch  = torch.from_numpy(test_pca).to(DEFAULT_DEVICE)    # (k, T)

# ---------- LISTA-SRC ---------- #
listasrc = SRCClassifierLISTA(
    depth=10,
    lam=0.1,
    learn_theta=False,          # 若想微调阈值，可设 True
    device=DEFAULT_DEVICE
)
listasrc.fit(train_torch, train_lbls)
preds = listasrc.predict(test_torch)

# ---------- 评估 ---------- #
acc = (preds == test_lbls).mean()
print(f"Test Accuracy via LISTA-SRC + PCA = {acc:.2%}")

cm = confusion_matrix(test_lbls, preds)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["guitar","flute","violin","clarinet","trumpet","cello","saxophone"]
)
fig, ax = plt.subplots(figsize=(8,6))
disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix for LISTA-SRC Classifier")
plt.tight_layout()
plt.savefig("confusion_matrix_lista.png")
plt.show()
