"""Supervised CT-Net fitting, then frozen-feature Kalman parameter estimation."""
import numpy as np
import torch
from core.ctnet import CNNTransformerRegressor
from core.kalman import fit_parameters


def train_encoder(window_sequences, label_sequences, epochs=200, batch_size=128, device='cpu'):
    windows = np.concatenate(window_sequences).astype(np.float32)
    labels = np.concatenate(label_sequences).astype(np.float32)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(windows), torch.from_numpy(labels)),
        batch_size=batch_size, shuffle=True)
    model = CNNTransformerRegressor(in_channels=8, output_dims=6).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    model.train()
    for _ in range(epochs):
        losses = []
        for window, label in loader:
            prediction = model(window.to(device))
            loss = torch.nn.functional.mse_loss(prediction, label.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        scheduler.step(float(np.mean(losses)))
    return model


def fit_decoder(model, ordered_window_sequences, ordered_label_sequences, device='cpu'):
    model.eval()
    model.requires_grad_(False)
    feature_sequences = []
    with torch.no_grad():
        for windows in ordered_window_sequences:
            features = []
            for batch in np.array_split(windows, max(1, int(np.ceil(len(windows)/128)))):
                tensor = torch.as_tensor(batch, dtype=torch.float32, device=device)
                features.append(model(tensor, return_feature=True).cpu().numpy())
            feature_sequences.append(np.concatenate(features))
    return fit_parameters(feature_sequences, ordered_label_sequences, dt=0.02)
