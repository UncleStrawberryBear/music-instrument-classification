from torch.utils.data import DataLoader
from dataset import AudioInstrumentDataset
from torch.optim import SGD
import torch.nn as nn
import numpy as np
from model import CNNInstrumentClassifier
from constants import *

sequence_length = int(SAMPLE_RATE*0.5)

train_data = AudioInstrumentDataset(r"train_metadata.csv", sequence_length=sequence_length)
# print(train_data[0])  # Debug

loader = DataLoader(train_data, batch_size=32, shuffle=True)

model = CNNInstrumentClassifier()

def train(loader, model, n_epochs=5):
    # Optimization
    opt = SGD(model.parameters(), lr=0.01)
    Loss = nn.CrossEntropyLoss()

    # Train model
    losses = []
    epochs = []
    for epoch in range(n_epochs):
        N = len(loader)
        for i, (x, y) in enumerate(loader):
            opt.zero_grad()
            loss_value = Loss(model(x), y)
            loss_value.backward()
            opt.step()

            losses.append(loss_value.item())
            epochs.append(epoch + i / N)
        print(f'Epoch {epoch}, Loss: {loss_value.item()}')
    return np.array(epochs), np.array(losses)


# %%
epoch_data, loss_data = train(loader, model)