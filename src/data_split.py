"""
RoseNet 2024
Sarah Coffland and Katie Christensen
Create the data splits for train, validation, and tests 1, 2, and 3 data sets.
"""

# Third party imports
import torch
import random
import numpy as np

# Local imports
from utils import load_config

"""
    TRAIN: 80% of what's left after creating Tests 2 and 3.
    DEV: 10% of what's left after creating Tests 2 and 3.
    TEST 1: Totally random singular locations (10% of what's left after creating Tests 2 and 3).
    TEST 2: Random slices of the same position with all amino acids (5% of total).
    TEST 3: Random slices of the same amino acids at all positions (5% of total).
"""

def create_test3(full_tensor, config):
    """
        Create Test3.
    """
    test3_tensor = torch.zeros([0,24])

    aas = ['A','R','N','D','C','E','Q','G','H','I','L','K','M','F','P','S','T','W','Y','V']

    while len(test3_tensor) < int(config.get("lower_bound")*90000):
        rand_idx = random.randint(0, 19)
        rand_amino_one = ord(aas[rand_idx])

        rand_idx = random.randint(0, 19)
        rand_amino_two = ord(aas[rand_idx])

        amino_pair_tensors = full_tensor[(full_tensor[:,20] == rand_amino_one) & (full_tensor[:,21] == rand_amino_two)]
        
        if len(test3_tensor) + len(amino_pair_tensors) <= int(config.get("upper_bound")*90000):
            test3_tensor = torch.cat((test3_tensor, amino_pair_tensors), dim=0)
        
        print("Residues: ", rand_amino_one, rand_amino_two)
    
    mask = np.zeros(len(full_tensor), dtype=bool)
    
    test3_np = test3_tensor.numpy()
    
    for i, datapoint in enumerate(full_tensor.numpy()):
        mask[i] = np.any(np.all(test3_np == datapoint, axis=1))
    
    rest_of_tensor = full_tensor[~mask]
    
    return test3_tensor, rest_of_tensor

def create_test2(full_tensor, config):
    """
        Create Test2.
    """
    test2_tensor = torch.zeros([0,24])

    while len(test2_tensor) < int(config.get("lower_bound")*90000):
        rand_pos_one = random.randint(1, config.get("protein_length")+1)
        rand_pos_two = random.randint(1, config.get("protein_length")+1)
        
        pos_tensors = full_tensor[(full_tensor[:,22] == rand_pos_one) & (full_tensor[:,22] == rand_pos_two)]
        
        if len(test2_tensor) + len(pos_tensors) <= int(config.get("upper_bound")*90000):
            test2_tensor = torch.cat((test2_tensor, pos_tensors), dim=0)
        
        print("Positions: ", rand_pos_one, rand_pos_two)
    
    mask = np.zeros(len(full_tensor), dtype=bool)
    
    test2_np = test2_tensor.numpy()
    
    for i, datapoint in enumerate(full_tensor.numpy()):
        mask[i] = np.any(np.all(test2_np == datapoint, axis=1))
    
    rest_of_tensor = full_tensor[~mask]
    rest_of_tensor = full_tensor[~mask]
    
    return test2_tensor, rest_of_tensor

def create_train_dev_test1(full_tensor):
    """
        Create Train/Validation/Test1.
    """
    indices = [x for x in range(0, len(full_tensor))]
    random.shuffle(indices)

    train_amt = int(len(full_tensor)*0.8)
    dev_amt =  int(len(full_tensor) * 0.1)

    train_indices = torch.tensor(indices[:train_amt])
    dev_indices = torch.tensor(indices[train_amt:train_amt + dev_amt])
    test_indices = torch.tensor(indices[train_amt + dev_amt:])

    train = full_tensor[train_indices]
    dev = full_tensor[dev_indices]
    test1 = full_tensor[test_indices]

    return train, dev, test1

def main():
    config = load_config("../config/data.yaml")

    full_data = torch.load(config.get("data_path"))
    save_path = config.get("save_path")

    test3, rest_of_tensor = create_test3(full_data, config)
    test2, rest_of_tensor = create_test2(rest_of_tensor, config)
    train, dev, test1 = create_train_dev_test1(rest_of_tensor, config)

    # Check that each of the tensors added together equals the full data
    if len(test1) + len(test2) + len(test3) + len(train) + len(dev) != len(full_data):
        print("Data leakage! Please check your splits.")
        exit(0)

    print("train: ", len(train))
    print("dev: ", len(dev))
    print("test1: ", len(test1))
    print("test2: ", len(test2))
    print("test3: ", len(test3))

    torch.save(test3, save_path + "test3.pt")
    torch.save(test2,  save_path + "test2.pt")
    torch.save(test1,  save_path + "test1.pt")
    torch.save(dev,  save_path + "dev.pt")
    torch.save(train,  save_path + "train.pt")

if __name__ == "__main__":
    main()