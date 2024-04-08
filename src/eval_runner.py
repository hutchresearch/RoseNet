"""
RoseNet 2024
Sarah Coffland and Katie Christensen
This file runs each of the epochs for the evaluation pipeline.
"""

# Third party imports
import wandb
import torch
import torchtext
from torch import nn
from tqdm import tqdm
from torch import save
from torch.optim import Optimizer
from contextlib import nullcontext
from torch.utils.data import DataLoader
from torchtext.data import get_tokenizer

# Local imports
from utils import make_one_to_one

class DevRunner:
    def __init__(
            self,
            dataloader: DataLoader,
            optimizer: Optimizer,
            model,
            device,
    ):
        self.dataloader = dataloader
        self.optimizer = optimizer

        self.criterion = nn.HuberLoss()
        self.model = model
        self.device = device

        model = model.to(device)


    def run_epoch(self, training=True):
        torch.cuda.empty_cache()

        self.model.eval()

        running_loss = 0.0

        self.optimizer.zero_grad()

        all_predictions = []
        all_targets = []
        all_inputs = []

        with torch.no_grad() if not training else nullcontext():
            for inputs, targets in tqdm(self.dataloader):

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
                all_inputs.append(inputs.detach().cpu().clone())
                
                inputs = None
                targets = None  
                prediction = None
                loss = None

        
        average_loss = running_loss / len(self.dataloader)

        return (average_loss), torch.vstack(all_targets), torch.vstack(all_predictions), torch.vstack(all_inputs)
    
    def save_state(self, path: str):
        if path is not None and path != '':
            save(self.model.state_dict(), path)
        

class TestRunner:
    def __init__(
            self,
            test1_loader: DataLoader,
            test2_loader: DataLoader,
            test3_loader: DataLoader,
            optimizer: Optimizer,
            model,
            device,
    ):
        self.test1_loader = test1_loader
        self.test2_loader = test2_loader
        self.test3_loader = test3_loader
        self.optimizer = optimizer

        self.criterion = nn.HuberLoss()
        self.model = model
        self.device = device

        model = model.to(device)

    def run_epoch(self, test, training=True):
        torch.cuda.empty_cache()

        self.model.eval()

        if test == 1:
            dataloader = self.test1_loader
        elif test == 2:
            dataloader = self.test2_loader
        elif test == 3:
            dataloader = self.test3_loader

        running_loss = 0.0

        self.optimizer.zero_grad()

        all_predictions = []
        all_targets = []
        all_inputs = []

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
                all_inputs.append(inputs.detach().cpu().clone())
                
                inputs = None
                targets = None  
                prediction = None
                loss = None

        average_loss = running_loss / len(dataloader)
        
        return (average_loss), torch.vstack(all_targets), torch.vstack(all_predictions), torch.vstack(all_inputs)
    
    def save_state(self, path: str):
        if path is not None and path != '':
            save(self.model.state_dict(), path)