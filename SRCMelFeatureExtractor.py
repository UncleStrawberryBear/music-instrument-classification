import torch
import torch.nn as nn
import torchaudio.transforms as T

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
        features = mel_db.view(B, -1)  # (B, M*T_)

        return features