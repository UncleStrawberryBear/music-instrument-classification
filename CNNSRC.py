import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from dataset import AudioInstrumentDataset
from model import CNNInstrumentClassifier
from src_classifier import SRCClassifier
from constants import SAMPLE_RATE, DEFAULT_DEVICE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Parameters
sequence_length = int(SAMPLE_RATE * 0.5)

# Load datasets
train_ds = AudioInstrumentDataset("train_metadata.csv", sequence_length)
test_ds = AudioInstrumentDataset("test_metadata.csv", sequence_length)

# Initialize CNN and freeze weights
cnn_model = CNNInstrumentClassifier(num_classes=7).to(DEFAULT_DEVICE)
cnn_model.eval()

# Helper: extract CNN features (before fc layer)
def extract_cnn_features(dataset):
    features, labels = [], []
    with torch.no_grad():
        for i in range(len(dataset)):
            waveform, onehot = dataset[i]  # waveform: [1, seq_len]
            waveform = waveform.unsqueeze(0)  # => (1, 1, seq_len)

            mel_spec = cnn_model.mel_spectrogram(waveform)
            log_spec = cnn_model.log_transform(mel_spec)
            cnn_input = log_spec  # (1, 1, M, T)

            cnn_feat = cnn_model.cnn(cnn_input)  # (1, 128, h, w)
            pooled = cnn_model.adaptive_pool(cnn_feat)  # (1, 128, 1, 1)
            feat_vec = pooled.view(-1).cpu().numpy()  # (128,)

            features.append(feat_vec)
            labels.append(onehot.argmax().item())

    features = np.stack(features, axis=1).astype(np.float32)  # (128, num_samples)
    return features, labels

# Extract CNN features
print("Extracting train features...")
train_features, train_labels = extract_cnn_features(train_ds)
print("Extracting test features...")
test_features, test_labels = extract_cnn_features(test_ds)

# Convert to tensors
train_tensor = torch.from_numpy(train_features).to(DEFAULT_DEVICE)
test_tensor = torch.from_numpy(test_features).to(DEFAULT_DEVICE)

# Run SRC
src_model = SRCClassifier(sparsity=20, device=DEFAULT_DEVICE)
src_model.fit(train_tensor, train_labels)
preds = src_model.predict(test_tensor).tolist()

# Accuracy
acc = sum(int(p == t) for p, t in zip(preds, test_labels)) / len(test_labels)
print(f"Test Accuracy via CNN+SRC = {acc:.2%}")

# Confusion Matrix
cm = confusion_matrix(test_labels, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[
    "guitar", "flute", "violin", "clarinet", "trumpet", "cello", "saxophone"])
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix for CNN+SRC Classifier")
plt.tight_layout()
plt.savefig("confusion_matrix_cnnsrc.png")
plt.show()
