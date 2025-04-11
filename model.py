import torch
import torch.nn as nn
import torchaudio.transforms as T
import torch.nn.functional as F
from torchaudio.transforms import FrequencyMasking, TimeMasking

from constants import *


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.Identity() if in_channels == out_channels else \
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
        
    def forward(self, x):
        return F.relu(self.conv(x) + self.shortcut(x))


class CNNInstrumentClassifier(nn.Module):
    def __init__(self, num_classes=4, sample_rate=SAMPLE_RATE, n_mels=128):
        super().__init__()

        # Audio preprocessing pipeline
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=256,
            n_mels=n_mels,
            normalized=True  # Enable built-in normalization
        )
        
        self.log_transform = T.AmplitudeToDB(stype="power")
        self.spec_augment = nn.Sequential(
            FrequencyMasking(freq_mask_param=15),
            TimeMasking(time_mask_param=35)
        )

        self.cnn = nn.Sequential(
            # Block 1
            ResidualBlock(1, 32),
            nn.MaxPool2d(2, 2),
            
            # Block 2
            ResidualBlock(32, 64),
            nn.MaxPool2d(2, 2),
            
            # Block 3
            ResidualBlock(64, 128),
            nn.MaxPool2d(2, 2),
            
            # Block 4
            ResidualBlock(128, 256),
            nn.AdaptiveAvgPool2d(1)
        )

        self.se = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 16),  # SE压缩通道
            nn.ReLU(inplace=True),
            nn.Linear(16, 256),  # SE激励通道
            nn.Sigmoid()
        )

        # Classifier
        self.fc = nn.Sequential(
            nn.Dropout(0.5),  # 防止过拟合
            nn.Linear(256, num_classes)
        )

    def forward(self, waveform):
        # Input normalization
        waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-8)
        
        # Feature extraction
        x = self.mel_spectrogram(waveform)
        x = self.log_transform(x)
        
        # Data augmentation (only during training)
        if self.training:
            x = self.spec_augment(x)
        
        # CNN processing with residual blocks
        x = x.unsqueeze(1)  # Add channel dim
        features = self.cnn(x)
        
        # SE attention reweighting
        b, c, _, _ = features.shape
        features_flat = features.view(b, c)
        se_weights = self.se(features)
        features_weighted = features_flat * se_weights
        
        # Classification
        return self.fc(features_weighted)