"""
RoseNet 2024
Sarah Coffland and Katie Christensen
This file contains utility functions used throughout the pipeline.
"""
# Standard library imports
import os

# Third party imports
import yaml
import torch 
import scipy
import numpy as np
from KDEpy import FFTKDE
import matplotlib.pyplot as plt

def make_one_to_one(config, true, predictions, title, kde_bandwidth=0.01, grid_size=2 ** 8, y_range=(-5,5), x_range=(-5,5)):
    """
        Creates one to one plots of the given data.
    """
    true_pred = torch.vstack([true,predictions]).permute(1,0)
    pred_probs = calc_KDE(true_pred, kde_bandwidth, grid_size)

    fig, ax = plt.subplots()

    plt.plot([-10000, 10000], [-10000, 10000], c="k", linewidth=1)
    
    one_to_one = ax.scatter(
        true, 
        predictions, 
        c=pred_probs,
        cmap="plasma", 
        s=1, 
        linewidths=0
    )

    cbar = fig.colorbar(one_to_one, ax=ax)
    cbar.set_label("density")

    plt.ylim(*y_range)
    plt.xlim(*x_range)

    plt.title(title)
    plt.xlabel("True", fontsize=15)
    plt.ylabel("Pred", fontsize=15)

    save_path = os.path.join(config.get('one_to_one_save_base_path'), title)
    plt.savefig(save_path)
    plt.close()
    return

def calc_KDE(labels: torch.Tensor, kde_bandwidth: float, grid_size) -> torch.Tensor:
    """
        This will calculate the pdf from a KDE of labels (Size([num_datapoints, num_labels])).
        Note: grid_size will significantly slow things down if it is too large. 2**8 works for a
        Nx2 array, but 2**3 works for an Nx3 array. Anything more may not be worth doing.
    """
    assert(len(labels.shape) <= 2), "Incorrect shape for labels tensor."

    if kde_bandwidth == None:
        kde_bandwidth=1

    labels = labels.numpy()
    labels_linear_grid, grid_probs = FFTKDE(kernel="gaussian", bw=kde_bandwidth).fit(labels, weights=None).evaluate(grid_size)

    if labels.shape[-1] == 1 or len(labels.shape) == 1:
        # For 1D arrays
        interp = scipy.interpolate.interp1d(labels_linear_grid, grid_probs)
        labels_probs = interp(labels)
    else:
        # For ND arrays
        interp = scipy.interpolate.LinearNDInterpolator(labels_linear_grid, grid_probs)
        labels_split = [label_type.squeeze() for label_type in np.split(labels, labels.shape[1], axis=1)]
        labels_probs = interp(*labels_split)

    return torch.from_numpy(labels_probs)

def load_config(config_file):
    """
        Reads the config yaml file and returns its contents.
    """
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    f.close()
    return config 