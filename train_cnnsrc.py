# train.py
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import AudioInstrumentDataset
from model import CNNInstrumentClassifier
from src_classifier import SRCClassifier
from constants import SAMPLE_RATE, DEFAULT_DEVICE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

sequence_length = int(SAMPLE_RATE * 0.5)
batch_size = 32
num_epochs = 20
learning_rate = 1e-3
num_classes = 7
sparsity = 20

train_ds = AudioInstrumentDataset("train_metadata.csv", sequence_length)
val_ds = AudioInstrumentDataset("test_metadata.csv", sequence_length)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

model = CNNInstrumentClassifier(num_classes=num_classes).to(DEFAULT_DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(1, num_epochs + 1):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for waveforms, onehots in train_loader:
        waveforms = waveforms.to(DEFAULT_DEVICE)  # (B,1,seq_len)
        labels = onehots.argmax(dim=1).to(DEFAULT_DEVICE)
        optimizer.zero_grad()
        outputs = model(waveforms)  # (B,num_classes)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / total
    train_acc = correct / total

    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for waveforms, onehots in val_loader:
            waveforms = waveforms.to(DEFAULT_DEVICE)
            labels = onehots.argmax(dim=1).to(DEFAULT_DEVICE)
            preds = model(waveforms).argmax(dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
    val_acc = val_correct / val_total
    print(
        f"Epoch {epoch}/{num_epochs}  "
        f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.3%}  "
        f"Val Acc: {val_acc:.3%}"
    )

torch.save(model.state_dict(), "cnn_classifier.pkl")
print("Saved CNN model to cnn_classifier.pkl")


# --- 2. 特征提取 (128-d) + SRC分类 ---
def extract_cnn_features(dataset, model):
    model.eval()
    feats, labels = [], []
    with torch.no_grad():
        for waveform, onehot in dataset:
            # waveform: (1, seq_len) -> (1,1,seq_len)
            wav = waveform.unsqueeze(0).to(DEFAULT_DEVICE)
            mel = model.mel_spectrogram(wav)
            log_mel = model.log_transform(mel)
            feat_map = model.cnn(log_mel)
            pooled = model.adaptive_pool(feat_map)
            vec = pooled.view(-1).cpu().numpy()  # (128,)
            feats.append(vec)
            labels.append(onehot.argmax().item())
    feats = np.stack(feats, axis=1).astype(np.float32)  # (128, N)
    return feats, labels


print("Extracting features...")
train_feats, train_labels = extract_cnn_features(train_ds, model)
test_feats, test_labels = extract_cnn_features(val_ds, model)

train_tensor = torch.from_numpy(train_feats).to(DEFAULT_DEVICE)
test_tensor = torch.from_numpy(test_feats).to(DEFAULT_DEVICE)

src = SRCClassifier(sparsity=sparsity, device=DEFAULT_DEVICE)
src.fit(train_tensor, train_labels)
preds = src.predict(test_tensor).tolist()

accuracy = sum(int(p == t) for p, t in zip(preds, test_labels)) / len(test_labels)
print(f"SRC Test Accuracy = {accuracy:.2%}")

cm = confusion_matrix(test_labels, preds)
disp = ConfusionMatrixDisplay(
    cm,
    display_labels=[
        "guitar",
        "flute",
        "violin",
        "clarinet",
        "trumpet",
        "cello",
        "saxophone",
    ],
)
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix for CNN+SRC Classifier")
plt.tight_layout()
plt.savefig("confusion_matrix_cnnsrc.png")
plt.show()
