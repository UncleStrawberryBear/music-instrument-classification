"""
The neural network model for the task
"""

import torch
import torch.nn as nn
import torchaudio.transforms as T

from constants import *

class CNNInstrumentClassifier(nn.Module):
    def __init__(self, num_classes=4, sample_rate=SAMPLE_RATE, n_mels=128):
        super().__init__()

        # Convert waveform to Mel Spectrogram
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=256,
            n_mels=n_mels,
        )

        self.log_transform = T.AmplitudeToDB(stype="power")  # Convert to log scale

        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Adaptive pooling to ensure a fixed output size
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)  # Output shape: (batch, channels, 1, 1)

        # Fully connected layer
        self.fc = nn.Linear(128, num_classes)  # Maps from 128 features to the number of classes

    def forward(self, waveform):
        # Convert waveform to mel spectrogram
        mel_spec = self.mel_spectrogram(waveform)  # Shape:
        mel_spec = self.log_transform(mel_spec)  # Convert to log scale. Shape:

        # CNN feature extraction
        features = self.cnn(mel_spec)  # Shape: (batch, 128, H, W)

        # Adaptive pooling
        features = self.adaptive_pool(features)  # Shape: (batch, 128, 1, 1)

        # Flatten for FC Layer
        features = features.view(features.size(0), -1)  # Shape: (batch, 128)

        # Classification Layer
        output = self.fc(features)  # Shape: (batch, num_classes)

        return output


from constants import SAMPLE_RATE, DEFAULT_DEVICE

class SRCMelFeatureExtractor(nn.Module):
    """
    仅做“waveform -> mel_spectrogram -> log幅度 -> flatten”
    不做卷积网络；后续交给 SRC 算法进行分类。
    """

    def __init__(self, sample_rate=SAMPLE_RATE, n_mels=128):
        super().__init__()
        # 与原 CNN 相同的超参
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=256,
            n_mels=n_mels
        ).to(DEFAULT_DEVICE)

        self.amplitude_to_db = T.AmplitudeToDB(stype='power').to(DEFAULT_DEVICE)

    def forward(self, waveform):
        """
        waveform: [B, 1, sequence_length]
        返回： [B, feat_dim] 的 2D 张量 (batch, mel_bins * time_frames)
        """
        # (B, 1, seq_len) -> (B, 1, n_mels, time_frames)
        mel_spec = self.mel_spectrogram(waveform)
        mel_db = self.amplitude_to_db(mel_spec)  # log scale

        # 展平： (B, 1, n_mels, T) -> (B, n_mels * T)
        B, _, M, T_ = mel_db.shape
        #features = mel_db.view(B, -1)  # (B, M*T_)
        features = mel_db.reshape(B, -1)

        return features
