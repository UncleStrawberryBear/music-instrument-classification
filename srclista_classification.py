import torch
import numpy as np
from dataset import AudioInstrumentDataset
from model import SRCMelFeatureExtractor
from constants import SAMPLE_RATE, DEFAULT_DEVICE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from listasrc import LISTA, SRCClassifierLISTA

# 1) 准备数据
sequence_length = int(SAMPLE_RATE * 0.5)
train_ds = AudioInstrumentDataset("train_metadata.csv", sequence_length)
test_ds = AudioInstrumentDataset("test_metadata.csv", sequence_length)

feature_extractor = SRCMelFeatureExtractor(sample_rate=SAMPLE_RATE).to(DEFAULT_DEVICE)

train_features = []
train_labels = []
for i in range(len(train_ds)):
    waveform, onehot = train_ds[i]
    feat_vec = feature_extractor(waveform.unsqueeze(0))
    train_features.append(feat_vec.squeeze(0).cpu().numpy())
    train_labels.append(onehot.argmax().item())
train_features = np.array(train_features, dtype=np.float32).T

test_features = []
test_labels = []
for i in range(len(test_ds)):
    waveform, onehot = test_ds[i]
    feat_vec = feature_extractor(waveform.unsqueeze(0))
    test_features.append(feat_vec.squeeze(0).cpu().numpy())
    test_labels.append(onehot.argmax().item())
test_features = np.array(test_features, dtype=np.float32).T
test_features = test_features[:, :20]
test_labels = test_labels[:20]

# 2) PCA 降维
train_norm = train_features / (np.linalg.norm(train_features, axis=0, keepdims=True) + 1e-8)
test_norm = test_features / (np.linalg.norm(test_features, axis=0, keepdims=True) + 1e-8)
cov_train = train_norm @ train_norm.T
eigvals, eigvecs = np.linalg.eigh(cov_train)
idx = np.argsort(eigvals)[::-1]
eigvecs = eigvecs[:, idx]
energy = np.cumsum(eigvals[idx]) / np.sum(eigvals)
k_pca = np.argmax(energy >= 0.999)
V_pca = eigvecs[:, :k_pca + 1]
train_features_pca = V_pca.T @ train_norm
test_features_pca = V_pca.T @ test_norm

# 3) 训练 LISTA 模型
train_features_torch = torch.from_numpy(train_features_pca).to(DEFAULT_DEVICE)
test_features_torch = torch.from_numpy(test_features_pca).to(DEFAULT_DEVICE)

D = train_features_torch
labels = np.array(train_labels)

lista_model = LISTA(D, depth=10, device=DEFAULT_DEVICE)
optimizer = torch.optim.Adam(lista_model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()
X_target = torch.eye(D.shape[1], device=DEFAULT_DEVICE)
Y_input = D @ X_target

for epoch in range(1):
    lista_model.train()
    optimizer.zero_grad()
    X_pred = lista_model(Y_input.T)
    loss = loss_fn(X_pred, X_target.T)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

# 4) 分类
classifier = SRCClassifierLISTA(lista_model, labels)
preds = classifier.predict(test_features_torch)

# 5) 准确率 + 混淆矩阵
correct = sum([int(p == t) for p, t in zip(preds, test_labels)])
acc = correct / len(test_labels)
print(f"Test Accuracy via LISTA + PCA = {acc:.2%}")

cm = confusion_matrix(test_labels, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[
    "guitar", "flute", "violin", "clarinet", "trumpet", "cello", "saxophone"
])
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix for SRC + LISTA Classifier")
plt.tight_layout()
plt.savefig("confusion_matrix_lista.png")
plt.show()
