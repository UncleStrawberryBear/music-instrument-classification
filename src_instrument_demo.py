# src_instrument_demo.py

import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from dataset import AudioInstrumentDataset
from model import SRCMelFeatureExtractor
from src_classifier import SRCClassifier
from constants import SAMPLE_RATE, DEFAULT_DEVICE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay #Newly added, for confusion matrix
import matplotlib.pyplot as plt #Newly added

# 例如，使用 0.5s 的音频
sequence_length = int(SAMPLE_RATE * 0.5)

# 1) 准备数据
train_ds = AudioInstrumentDataset("train_metadata.csv", sequence_length)
test_ds = AudioInstrumentDataset("test_metadata.csv", sequence_length)

# 如果想看一下有多少条数据：
# print(len(train_ds), len(test_ds))

# 2) 我们的特征提取器
feature_extractor = SRCMelFeatureExtractor(sample_rate=SAMPLE_RATE).to(DEFAULT_DEVICE)

# 3) 把训练集所有样本转换为特征向量，并组装成字典
train_features = []
train_labels = []

for i in range(len(train_ds)):
    waveform, onehot = train_ds[i]  # waveform: [1, seq_len], onehot: [num_classes]
    feat_vec = feature_extractor(waveform.unsqueeze(0))  # 加一个batch维度 => (1, feat_dim)
    # feat_vec.shape = (1, mel_bins * time_frames)
    train_features.append(feat_vec.squeeze(0).cpu().numpy())
    label_idx = onehot.argmax().item()
    train_labels.append(label_idx)

train_features = np.array(train_features, dtype=np.float32).T  # => (feat_dim, num_train_samples)
train_features_torch = torch.from_numpy(train_features).to(DEFAULT_DEVICE)

# 4) 同理，处理测试集
test_features = []
test_labels = []

for i in range(len(test_ds)):
    waveform, onehot = test_ds[i]
    feat_vec = feature_extractor(waveform.unsqueeze(0))
    test_features.append(feat_vec.squeeze(0).cpu().numpy())
    label_idx = onehot.argmax().item()
    test_labels.append(label_idx)

test_features = np.array(test_features, dtype=np.float32).T  # (feat_dim, num_test_samples)
test_features_torch = torch.from_numpy(test_features).to(DEFAULT_DEVICE)

# 5) 用 SRCClassifier 做分类
src_model = SRCClassifier(sparsity=20, device=DEFAULT_DEVICE)
src_model.fit(train_features_torch, train_labels)

preds = src_model.predict(test_features_torch)

# 6) 计算准确率
preds = preds.tolist()
correct = 0
for p, t in zip(preds, test_labels):
    if p == t:
        correct += 1
acc = correct / len(test_labels)
print(f"Test Accuracy via SRC = {acc:.2%}")

cm = confusion_matrix(test_labels, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[
    "guitar", "flute", "violin", "clarinet", "trumpet", "cello", "saxophone"
])
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix for SRC Classifier")
plt.show()


