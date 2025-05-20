import torch
import numpy as np
from dataset import AudioInstrumentDataset
from model import SRCMelFeatureExtractor
from constants import SAMPLE_RATE, DEFAULT_DEVICE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from Lista_classifier import LISTA, SRCClassifierLISTA

# 设置参数
sequence_length = int(SAMPLE_RATE * 0.5)

# 1. 加载数据
train_ds = AudioInstrumentDataset("train_metadata.csv", sequence_length)
test_ds = AudioInstrumentDataset("test_metadata.csv", sequence_length)

# 2. 提取特征
feature_extractor = SRCMelFeatureExtractor(sample_rate=SAMPLE_RATE).to(DEFAULT_DEVICE)

def extract_features(dataset):
    features, labels = [], []
    for i in range(len(dataset)):
        waveform, onehot = dataset[i]
        feat_vec = feature_extractor(waveform.unsqueeze(0))
        features.append(feat_vec.squeeze(0).cpu().numpy())
        labels.append(onehot.argmax().item())
    features = np.array(features, dtype=np.float32).T  # (feat_dim, num_samples)
    return features, labels

train_features, train_labels = extract_features(train_ds)
test_features, test_labels = extract_features(test_ds)

# 3. PCA 降维
train_norm = train_features / (np.linalg.norm(train_features, axis=0, keepdims=True) + 1e-8)
test_norm = test_features / (np.linalg.norm(test_features, axis=0, keepdims=True) + 1e-8)
cov_train = train_norm @ train_norm.T
eigvals, eigvecs = np.linalg.eigh(cov_train)
idx = np.argsort(eigvals)[::-1]
eigvecs = eigvecs[:, idx]
energy = np.cumsum(eigvals[idx]) / np.sum(eigvals)
k_pca = np.argmax(energy >= 0.99)
V_pca = eigvecs[:, :k_pca + 1]
train_pca = V_pca.T @ train_norm
test_pca = V_pca.T @ test_norm

train_tensor = torch.from_numpy(train_pca).to(DEFAULT_DEVICE)
test_tensor = torch.from_numpy(test_pca).to(DEFAULT_DEVICE)

# 4. 创建并训练 LISTA
lista = LISTA(D=train_tensor, depth=10, device=DEFAULT_DEVICE)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(lista.parameters(), lr=1e-3)

X_target = train_tensor
Y_input = lista.D @ X_target

for epoch in range(30):
    lista.train()
    optimizer.zero_grad()
    X_pred = lista(Y_input.T)
    loss = loss_fn(X_pred, X_target.T)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

# 5. SRC 分类 + 评估
classifier = SRCClassifierLISTA(lista_model=lista, labels=np.array(train_labels))
preds = classifier.predict(test_tensor)

acc = np.mean(preds == np.array(test_labels))
print(f"Test Accuracy via SRC + LISTA + PCA = {acc:.2%}")

# 6. 混淆矩阵
cm = confusion_matrix(test_labels, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[
    "guitar", "flute", "violin", "clarinet", "trumpet", "cello", "saxophone"
])
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix for SRC + LISTA + PCA")
plt.tight_layout()
plt.savefig("confusion_matrix_lista.png")
plt.show()
