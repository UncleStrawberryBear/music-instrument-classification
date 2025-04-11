import torch
import torchaudio
import random
import os
from torch.utils.data import Dataset
import pandas as pd

from constants import *

class AudioInstrumentDataset(Dataset):
    def __init__(self, metadata_path, sequence_length, seed=42, device=DEFAULT_DEVICE):
        self.metadata = pd.read_csv(metadata_path)
        self.sequence_length = sequence_length
        self.random = random.Random(seed)
        self.device = device

        # Instrument mapping
        self.instrument_map = {"guitar": 0, "flute": 1, "violin": 2, "clarinet": 3}
        self.num_classes = len(self.instrument_map)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        """Returns (waveform, label) where waveform shape: [sequence_length]"""
        audio_path, instrument = self.metadata.iloc[idx]
        
        # Path handling
        audio_path = os.path.abspath(audio_path.replace('\\', '/'))

        # Load audio (shape: [channels, samples])
        waveform, sr = torchaudio.load(audio_path)

        # Resample if needed
        if sr != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=SAMPLE_RATE)

        # Convert to mono (shape: [samples])
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze(0)  # Force shape [samples]

        # Extract/pad segment
        if waveform.numel() > self.sequence_length:
            start = self.random.randint(0, waveform.numel() - self.sequence_length)
            waveform = waveform[start:start + self.sequence_length]
        else:
            waveform = torch.nn.functional.pad(
                waveform, 
                (0, max(0, self.sequence_length - waveform.numel()))
            )
        
        # # Data augmentation (optional)
        # if self.training:
        #     waveform = self._apply_augmentations(waveform)

        # Label conversion
        label = torch.tensor(self.instrument_map[instrument], dtype=torch.long)
        
        return waveform.to(self.device), label.to(self.device)

    def _apply_augmentations(self, waveform):
        """Optional time-domain augmentations"""
        # 1. Random gain
        if random.random() > 0.5:
            waveform = waveform * random.uniform(0.8, 1.2)
        
        # 2. Additive noise
        if random.random() > 0.7:
            noise = torch.randn_like(waveform) * 0.01
            waveform += noise
        
        return waveform.clamp(-1, 1)  # Maintain valid audio range