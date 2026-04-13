"""One-window inference from the CT-Net encoder to a compliant hand reference."""
import numpy as np
import torch
from core.admittance import DualReferenceAdmittance
from core.kalman import AdaptiveKalmanDecoder


class IntentPipeline:
    def __init__(self, encoder, parameters, device='cpu'):
        self.encoder = encoder.to(device).eval()
        self.device = device
        self.decoder = AdaptiveKalmanDecoder(parameters, alpha=0.05)
        self.controller = DualReferenceAdmittance()

    @torch.no_grad()
    def step(self, emg_window, external_force, visual_anchor=None):
        tensor = torch.as_tensor(emg_window, dtype=torch.float32, device=self.device)
        if tuple(tensor.shape) != (8, 200):
            raise ValueError("Expected one 8-channel, 200-sample EMG window.")
        feature = self.encoder(tensor.unsqueeze(0), return_feature=True)[0].cpu().numpy()
        intent = self.decoder.update(feature, visual_anchor)
        command = self.controller.update(intent[:3], intent[3:], external_force)
        return {"angle": intent[:3], "force": intent[3:], "command": command}

    def reset(self):
        self.decoder.reset()
        self.controller.reset()
