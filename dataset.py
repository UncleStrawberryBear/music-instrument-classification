"""
The Dataset subclass for this task
"""

import torch
import torchaudio
import random
import os
from torch.utils.data import Dataset
import pandas as pd

from constants import *

class AudioInstrumentDataset(Dataset):
    def __init__(self, metadata_path, sequence_length, seed = 42, device = DEFAULT_DEVICE):
        # self.metadata = pd.read_csv(metadata_path)
        # self.metadata["audio_path"] = self.metadata["audio_path"].apply(lambda p: p.replace("\\\\", "/"))
        # self.sequence_length = sequence_length
        # self.random = random.Random(seed)
        # self.device = device

        # # Define instrument-to-one-hot mapping
        # self.instrument_map = {"guitar": 0, "flute": 1, "violin": 2, "clarinet": 3}
        # self.num_classes = len(self.instrument_map)
        self.metadata = pd.read_csv(metadata_path)

        # Replace backslashes with forward slashes in ALL string columns (especially audio_path)
        for col in self.metadata.columns:
            if self.metadata[col].dtype == object:
                self.metadata[col] = self.metadata[col].apply(lambda x: x.replace("\\", "/") if isinstance(x, str) else x)

        self.sequence_length = sequence_length
        self.random = random.Random(seed)
        self.device = device

        # Define instrument-to-one-hot mapping
        self.instrument_map = {"guitar": 0, "flute": 1, "violin": 2, "clarinet": 3, " trumpet":4, "cello":5, "saxophone":6}
        self.num_classes = len(self.instrument_map)


    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        """
        Load an audio file, extract a random clip, and return (audio_clip, one_hot_label).
        """
        # Get the audio file path and instrument label
        audio_path, instrument = self.metadata.iloc[idx]

        # Load the audio file
        waveform, sr = torchaudio.load(audio_path)

        # Ensure the sample rate matches expected sample rate
        if sr != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=SAMPLE_RATE)

        # Convert stereo to mono (if needed)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Randomly extract a clip of length sequence_length
        num_samples = waveform.shape[1]
        if num_samples > self.sequence_length:
            # The ideal case, extract the clip directly
            start = self.random.randint(0, num_samples - self.sequence_length)
            waveform = waveform[:, start:start + self.sequence_length]
        else:
            # Pad if the audio is shorter than required
            pad_length = self.sequence_length - num_samples
            waveform = torch.nn.functional.pad(waveform, (0, pad_length))

        # Convert instrument name to one-hot encoding
        instrument_index = self.instrument_map[instrument]
        one_hot_label = torch.zeros(self.num_classes)
        one_hot_label[instrument_index] = 1

        # Move to device before returning
        waveform =waveform.to(self.device)
        one_hot_label = one_hot_label.to(self.device)

        return waveform, one_hot_label