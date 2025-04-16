import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from dataset import AudioInstrumentDataset
from model import SRCMelFeatureExtractor
from src_classifier import SRCClassifier
from constants import SAMPLE_RATE, DEFAULT_DEVICE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 例如，使用 0.5s 的音频
sequence_length = int(SAMPLE_RATE * 0.5)

# 1) 准备数据
train_ds = AudioInstrumentDataset("train_metadata.csv", sequence_length)
test_ds = AudioInstrumentDataset("test_metadata.csv", sequence_length)

# 2) 特征提取器
feature_extractor = SRCMelFeatureExtractor(sample_rate=SAMPLE_RATE).to(DEFAULT_DEVICE)

# 3) 提取训练集特征
train_features = []
train_labels = []
for i in range(len(train_ds)):
    waveform, onehot = train_ds[i]
    feat_vec = feature_extractor(waveform.unsqueeze(0))
    train_features.append(feat_vec.squeeze(0).cpu().numpy())
    label_idx = onehot.argmax().item()
    train_labels.append(label_idx)
train_features = np.array(train_features, dtype=np.float32).T  # (feat_dim, num_train)

# 4) 提取测试集特征
test_features = []
test_labels = []
for i in range(len(test_ds)):
    waveform, onehot = test_ds[i]
    feat_vec = feature_extractor(waveform.unsqueeze(0))
    test_features.append(feat_vec.squeeze(0).cpu().numpy())
    label_idx = onehot.argmax().item()
    test_labels.append(label_idx)
test_features = np.array(test_features, dtype=np.float32).T  # (feat_dim, num_test)

# 5) 使用 PCA 降维
# Normalize
train_norm = train_features / (np.linalg.norm(train_features, axis=0, keepdims=True) + 1e-8)
test_norm = test_features / (np.linalg.norm(test_features, axis=0, keepdims=True) + 1e-8)

# Covariance and eigendecomposition
cov_train = train_norm @ train_norm.T
eigvals, eigvecs = np.linalg.eigh(cov_train)
idx = np.argsort(eigvals)[::-1]
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]

# Select top-k to preserve 99.9% energy
energy = np.cumsum(eigvals) / np.sum(eigvals)
k_pca = np.argmax(energy >= 0.999)
V_pca = eigvecs[:, :k_pca + 1]

# Apply projection
train_features_pca = V_pca.T @ train_norm
test_features_pca = V_pca.T @ test_norm

# Convert to tensor
train_features_torch = torch.from_numpy(train_features_pca).to(DEFAULT_DEVICE)
test_features_torch = torch.from_numpy(test_features_pca).to(DEFAULT_DEVICE)

# 6) 训练 + 预测
src_model = SRCClassifier(sparsity=20, device=DEFAULT_DEVICE)
src_model.fit(train_features_torch, train_labels)
preds = src_model.predict(test_features_torch).tolist()

# 7) 准确率计算
correct = sum([int(p == t) for p, t in zip(preds, test_labels)])
acc = correct / len(test_labels)
print(f"Test Accuracy via SRC + PCA = {acc:.2%}")

# 8) 混淆矩阵可视化
cm = confusion_matrix(test_labels, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[
    "guitar", "flute", "violin", "clarinet", "trumpet", "cello", "saxophone"
])
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix for SRC + PCA Classifier")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()
