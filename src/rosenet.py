"""
RoseNet 2024
Sarah Coffland and Katie Christensen
Architectures of the RoseNet model and RoseNetBlock.
"""

# Third party imports
import torch
from torch import nn

class RoseNetBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(num_features=800)
        self.batch_norm_2 = nn.BatchNorm1d(num_features=800)
        self.activation = nn.ReLU()
        self.layer1 = nn.Linear(800, 800)
        self.layer2 = nn.Linear(800, 800)

    def forward(self, x):  
        h = x
        x = self.layer1(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.layer2(x)
        x = self.batch_norm_2(x)
        x = self.activation(x)
        x = h + x
        return x

class RoseNet(nn.Module):
    def __init__(self, scores_size: int, batch_size: int, protein_length: int, num_blocks):
        super().__init__()

        self.batch_size = batch_size
        self.protein_length = protein_length

        if self.protein_length > 110 or protein_length == 85: num_embeddings = 150 # 2ckx, 1c44, 5cvz
        else: num_embeddings = 110 # 1hhp, 1crn, 1csp

        self.identity_embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=200)
        self.position_embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=200)

        self.blocks = nn.ModuleList([RoseNetBlock() for _ in range(num_blocks)])
        self.linear = nn.Linear(800, scores_size)

    def forward(self, x):
        embedded = []
        embedded.append(self.position_embedding(torch.Tensor.int(x[:,0]))) # Position of insertion 1
        embedded.append(self.identity_embedding(torch.Tensor.int(x[:,1]))) # Residue identity of insertion 1
        embedded.append(self.position_embedding(torch.Tensor.int(x[:,2]))) # Position of insertion 2
        embedded.append(self.identity_embedding(torch.Tensor.int(x[:,3]))) # Residue identity of insertion 2
        
        x = torch.hstack(embedded)

        for block in self.blocks:
            x = block(x)
        
        x = self.linear(x)
        
        return x

if __name__ == '__main__':
    model = RoseNet(4, 64, 20, 1)