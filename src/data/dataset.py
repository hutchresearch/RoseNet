"""
RoseNet 2024
Sarah Coffland and Katie Christensen
A PyTorch Dataset for pdb data.
"""

# Third party imports
import torch
from pathlib import Path
from torch.utils.data import Dataset

def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    return device

class PDBDataset(Dataset):
    def __init__(
            self, 
            input_path: Path
            ):
        
        self.input_path = input_path
        self.input_matrix = torch.load(input_path)

        for score in range(20):
            val = self.input_matrix[:,score]
            if torch.std(val) != 0: self.input_matrix[:,score] = (val - torch.mean(val)) / torch.std(val)

    def __len__(self):
        return len(self.input_matrix)
    
    def estimate_num_batches(self, batch_size: int):
        num_batches = len(self) // batch_size

        if len(self) % num_batches != 0:
            num_batches += 1

        return num_batches

    def __getitem__(self, idx: int):   
        input_tensor = torch.zeros([4], dtype=torch.int32)

        input_tensor[1] = self.input_matrix[idx][20] # Residue identity of insertion 1
        input_tensor[3] = self.input_matrix[idx][21] # Residue identity of insertion 2 
        input_tensor[0] = self.input_matrix[idx][22] # Position of insertion 1
        input_tensor[2] = self.input_matrix[idx][23] # Position of insertion 2  

        scores_tensor = torch.cat((torch.cat((self.input_matrix[idx][:12], self.input_matrix[idx][13:16]), dim=0), self.input_matrix[idx][17:20]),dim=0)
        
        return input_tensor, scores_tensor