"""
RoseNet 2024
Sarah Coffland and Katie Christensen
Pipeline for training and evaluating the model.
"""

# Third party imports
import torch
from torch.optim import NAdam
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Local imports
from runner import Runner
from rosenet import RoseNet
from data.dataset import PDBDataset
from utils import make_one_to_one, load_config

def main():
    config = load_config("../config/main.yaml")
    
    scores = config.get('scores')

    train_set = PDBDataset(config.get("train_path"))
    dev_set = PDBDataset(config.get("dev_path"))

    # Initialize dataloaders
    train_loader = DataLoader(train_set, batch_size=config.get("batch_size"), shuffle=True, num_workers=config.get("num_workers"))
    dev_loader = DataLoader(dev_set, batch_size=config.get("batch_size"), shuffle=False, num_workers=config.get("num_workers"), pin_memory=False)

    # Create the model
    model = RoseNet(config.get("output_size"), batch_size=config.get("batch_size"), protein_length=config.get("protein_length"), num_blocks=config.get("num_blocks"))
    
    # Create the optimizer
    optimizer = NAdam(model.parameters(), lr=config.get("lr"))

    # Create the learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, 'min')

    if torch.cuda.is_available():
        device = torch.device("cuda") #uses GPU
    else:
        device = torch.device("cpu") #uses CPU
    
    # Create the runner
    runner = Runner(
        train_loader=train_loader,
        dev_loader=dev_loader,
        optimizer=optimizer,
        model=model,
        device=device,
    )

    # Keep track of the best validation loss
    best_dev_loss = 1000000

    # Train the model
    for epoch in range(config.get("epochs")):
        print("Epoch ", str(epoch), "/", str(config.get("epochs")), "")

        # Calculate train loss and R^2
        train_loss, train_targets, train_predictions = runner.run_epoch(training=True)
        train_r_2 = r2_score(train_targets, train_predictions)
        print("\nTrain loss: ", train_loss.item())
        print("R Squared: ", train_r_2, "\n")

         # Calculate validation loss and R^2
        dev_loss, dev_targets, dev_predictions = runner.run_epoch(training=False)
        dev_r_2 = r2_score(dev_targets, dev_predictions)
        print("Dev loss: ", dev_loss.item())
        print("R Squared: ", dev_r_2, "\n")

        # Update and save the model
        if dev_loss < best_dev_loss and not config.get("DEBUG"):
            if (epoch + 1) % 1 == 0:
                best_dev_loss = dev_loss
                runner.save_state(config.get("save_path"))

        # Take a step
        scheduler.step(train_loss)
        
        # Calculate the R^2 and Pearson correlation coefficients of the validation predictions for each Rosetta score
        for i, score in enumerate(scores):
            rosetta_score_r2 = r2_score(dev_targets[:,i], dev_predictions[:,i])
            pearson_correlation, p_value = pearsonr(dev_targets[:,i], dev_predictions[:,i])
            
            print(f"{score}_r2: {rosetta_score_r2}, {score}_pearson_correlation: {pearson_correlation}, p_value: {p_value} \n")
            
            # Create and save one to one plots every five epochs
            if epoch % 5 == 0:
                make_one_to_one(config, dev_targets[:,i], dev_predictions[:,i], f"{score}_one_to_one")

    return

if __name__ == "__main__":
    main()
