from __future__ import annotations

import torch
from tqdm import tqdm


class Tester:
    def __init__(self, model, dataloader, logger, device):
        self.model = model
        self.dataloader = dataloader
        self.logger = logger
        self.device = device

    @torch.no_grad()
    def test(self):
        self.model.eval()
        bar = tqdm(self.dataloader, desc="testing", leave=True)
        for inputs, targets, info in bar:
            inputs = inputs.to(self.device)
            _ = self.model(inputs)
        self.logger.info("testing finished")
