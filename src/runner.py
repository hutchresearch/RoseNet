"""
RoseNet 2024
Sarah Coffland and Katie Christensen
This file runs each of the epochs for the training pipeline.
"""

# Third party imports
import torch
from torch import nn
from tqdm import tqdm
from torch import save
from torch.optim import Optimizer
from contextlib import nullcontext
from torch.utils.data import DataLoader

class Runner:
    def __init__(
            self,
            train_loader: DataLoader,
            dev_loader: DataLoader,
            optimizer: Optimizer,
            model,
            device,
    ):
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.optimizer = optimizer

        self.criterion = nn.HuberLoss()
        self.model = model
        self.device = device

        model = model.to(device)


    def run_epoch(self, training=True):
        torch.cuda.empty_cache()

        self.model.train() if training else self.model.eval()
        
        dataloader = self.train_loader if training else self.dev_loader

        running_loss = 0.0

        self.optimizer.zero_grad()

        all_predictions = []
        all_targets = []

        with torch.no_grad() if not training else nullcontext():
            for inputs, targets in tqdm(dataloader):
                
                inputs, targets = (
                    inputs.to(self.device),
                    targets.to(self.device)
                )

                prediction = self.model(inputs.float())

                loss = self.criterion(prediction, targets)
                running_loss += loss

                if training:
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                all_predictions.append(prediction.detach().cpu().clone())
                all_targets.append(targets.detach().cpu().clone())
                
                inputs = None
                targets = None  
                prediction = None
                loss = None
        
        average_loss = running_loss / len(dataloader)

        return (average_loss), torch.vstack(all_targets), torch.vstack(all_predictions)
    
    def save_state(self, path: str):
        if path is not None and path != '':
            save(self.model.state_dict(), path)