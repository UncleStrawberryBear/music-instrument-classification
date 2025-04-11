import torch

SAMPLE_RATE = 16000

DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'