"""
RoseNet 2024
Sarah Coffland and Katie Christensen
Generates the solvent accessible surface area (SASA) and secondary structure masks used for evaluation.
"""

# Standard library imports
import os

# Third party imports
import torch
import subprocess
import numpy as np
import pandas as pd

# Local imports
from utils import load_config

def download_pdb_file(pdb_id, config):
    """
        Downloads the pdb file of the given pdb ID to the download directory.
        Returns: Boolean (True = downloaded, False = error/did not download)
    """
    # Creat the URL of the pdb file.
    pdb_file_URL = 'https://files.rcsb.org/header/' + pdb_id + '.pdb'

    # Create the path to download the files in.
    download_path = os.path.join(config.get("download_path"), pdb_id + '.pdb')

    # Check if the file already exists.
    if os.path.exists(download_path): return True
    
    # Create an empty file at the specified path (wget requires the file already exsits).
    with open(download_path, 'w') as f: 
        pass
    f.close()
    
    # Download the file.
    try:
        subprocess.run(['wget', '-O', download_path, pdb_file_URL], check=True)
    except subprocess.CalledProcessError as e:
        print(f'Failed to download the pdb file. Error: {e}')
        return False
    
    return True

def get_SS_location(line, type):
    """
        This function takes a line from the pdb file, parses the line, and finds the start and end locations
        of the alpha helix or beta sheet (specified in the parameter 'type'). 
    """
    # Split the lines
    tokens = line.split()

    # Initialize the start and end locations of the secondary structures.
    start=0
    end=0

    # Some lines for beta sheets contain multiple start/end locations.
    start2 = 0
    end2 = 0
    
    if tokens[0].lower() == type.lower():
        # Grab the tokens at the 5th and 8th indicies for lines specifying alpha helices.
        if type.lower() == 'helix':
            start = int(tokens[5])
            end = int(tokens[8])

        # Grab the tokens at the 6th and 6th indicies for lines specifying beta sheets.
        elif type.lower() == 'sheet':
            start = int(tokens[6])
            end = int(tokens[9])

            # If this line contains multiple start/end locations, also grab the 14th and 18th tokens.
            if tokens[10] == '-1' and len(tokens) > 11:
                start2 = int(tokens[14])
                end2 = int(tokens[18])

    return start, end, start2, end2

def generate_secondary_structure_mask(pdb_id, protein_length, config):
    """
        This function creates the mask for the secondary structures. 
        In the mask, 0 denotes no secondary structure at that location,
        1 represents an alpha helix at the location, and 2 represents a beta sheet.  
    """
    # Initialize the mask as the length of the wild type.
    mask = np.zeros(protein_length)

    # If we need to download the pdb files (e.g., they don't already exist in the filesystem), download them.
    if config.get("download") == False: download_successful = download_pdb_file(pdb_id=pdb_id, config=config)
    else: download_successful = True

    if download_successful == False:
        print("Error downloading pdb file: ", pdb_id)
        exit()
    else:
        # Create the path to the downloaded pdb file.
        file_path = os.path.join(config.get("download_path"), f"{pdb_id}.pdb")

        # Read the file.
        with open(file_path, 'r') as f:
            pdb_content_lines = f.readlines()
        f.close()

        for line in pdb_content_lines:

            # Find where the file describes the alpha helices.
            if 'HELIX' in line: 
                # Find the start and end locations of the current alpha helix.
                start, end, start2, end2 = get_SS_location(line,'helix')

                if start != 0 and end != 0:
                    # PDB ID 2ckx starts at location 577 instead of 1, subtract that to get the indicies for our mask.
                    if pdb_id.lower() == '2ckx':
                        start = start-577
                        end = end-577
                        if start2 != 0 and end2 != 0:
                            start2 = start2-577
                            end2 = end2-577

                    # Iterate over the mask, adding 1's at the locations with the alpha helix.
                    for i, _ in enumerate(mask):
                        if i < start-1: continue 

                        if start-1 <= i <= end-1: mask[i] = 1

                        if start2 == 0 and end2 == 0:
                            if i == end: break 

                        if start2 != 0 and end2 != 0:
                            if end-1 < i < start2-1: continue 
                            if start2-1 <= i <= end2-1: mask[i] = 1
                            if i == end2: break

            # Find where the file describes the beta sheets.
            if 'SHEET' in line: 
                # Find the start and end locations of the current beta sheet.
                start, end, start2, end2 = get_SS_location(line,'sheet')

                # PDB ID 2ckx does not contain beta sheets (we don't need to check for it and subtract 577 like above).

                # Iterate over the mask, adding 2's at the locations with the beta sheet.
                if start != 0 and end != 0:
                    for i, _ in enumerate(mask):
                        if i < start-1: continue 

                        if start-1 <= i <= end-1: mask[i] = 2

                        if start2 == 0 and end2 == 0:
                            if i == end: break 

                        if start2 != 0 and end2 != 0:
                            if end-1 < i < start2-1: continue 
                            if start2-1 <= i <= end2-1: mask[i] = 2
                            if i == end2: break
    
    return mask, f'{pdb_id}_mask'

def generate_original_SASA_mask(pdb_id, protein_length, config):
    """
        This function creates the 'original' SASA mask. Original refers to how we categorize low, medium, and high bins.
        In the mask, 0 = low. 1 = medium. 2 = high.
        Low: 0-33% rSASA/Volume. Medium: 31-66% rSASA/Volume. High: 67 - 100% rSASA/Volume 
    """
    # Initialize the mask as the length of the wild type.
    mask = np.zeros(protein_length)

    # Create a dictionary containing volumes of each of the 20 naturally occurring amino acids, found here: https://www.imgt.org/IMGTeducation/Aide-memoire/_UK/aminoacids/abbreviation.html
    residue_sizes = {'ALA':88.6, 'ARG':173.4, 'ASN':114.1, 'ASP':111.1, 'CYS':108.5, 'GLN':143.8, 'GLU':138.4, 'GLY':60.1, 'HIS':153.2, 'ILE':166.7, 'LEU':166.7, 'LYS':168.6, 
                        'MET':162.9, 'PHE': 189.9, 'PRO':112.7, 'SER':89.0, 'THR':116.1, 'TRP':227.8, 'TYR':193.6, 'VAL':140.0}
    
    # Create the path to the saved text file of SASA information per residue, created using PyMol.
    sasa_path = os.path.join(config.get("SASA_path"), f"{pdb_id.lower()}.txt")

    # Sort the text file based on residue position. 
    sort_SASA(sasa_path)

    with open(sasa_path, "r") as f:
        lines = f.readlines()
    f.close()

    for i, line in enumerate(lines):
        tokens = line.split()

        # Grab the SASA value at the last token of each line.
        SASA = float(tokens[-1])

        # Grab the residue at the 3rd index of each line.
        res = tokens[3]
        
        # Looks like MET is the parent of MSE, so using MET's volume here for simplicity.
        if res == 'MSE': res = 'MET'
        
        # Calculate the SASA/volume.
        percentage = SASA/residue_sizes[res]

        # Add the labels to the mask.
        if percentage <= 0.33: mask[i]=0
        elif percentage > 0.33 and percentage <= 0.66: mask[i] = 1
        else: mask[i] = 2
    
    return mask, f'{pdb_id}_mask'

def generate_percentile_SASA_mask(pdb_id, protein_length, config):
    """
        This function creates the 'percentile' SASA mask. Percentile refers to how we categorize low, medium, and high bins.
        In the mask, 0 = low. 1 = medium. 2 = high.
        Low: x<=33rd percentile of SASA/volume.  Medium: 33rd<x<=66th percentile.  High: x>66th percentile,
        where x refers to the current residue. 
    """
    # Initialize the mask as the length of the wild type.
    mask = np.zeros(protein_length)

    # Create a dictionary containing volumes of each of the 20 naturally occurring amino acids, found here: https://www.imgt.org/IMGTeducation/Aide-memoire/_UK/aminoacids/abbreviation.html
    residue_sizes = {'ALA':88.6, 'ARG':173.4, 'ASN':114.1, 'ASP':111.1, 'CYS':108.5, 'GLN':143.8, 'GLU':138.4, 'GLY':60.1, 'HIS':153.2, 'ILE':166.7, 'LEU':166.7, 'LYS':168.6, 'MET':162.9, 'PHE': 189.9, 'PRO':112.7, 'SER':89.0, 'THR':116.1, 'TRP':227.8, 'TYR':193.6, 'VAL':140.0}
    
    # Create the path to the saved text file of SASA information per residue, created using PyMol.
    sasa_path = os.path.join(config.get("SASA_path"), f"{pdb_id.lower()}.txt")

    # Sort the text file based on residue position.
    sort_SASA(sasa_path)

    with open(sasa_path, "r") as f:
        lines = f.readlines()
    f.close()

    # Calculate the 33rd and 66th percentiles of the SASA/volumes. 
    percentile_33rd, percentile_66th = calc_SASA_percentiles(lines, residue_sizes)

    for i, line in enumerate(lines):
        tokens = line.split()

        # Grab the SASA value at the last token of each line.
        SASA = float(tokens[-1])

        # Grab the residue at the 3rd index of each line.
        res = tokens[3]

        # Looks like MET is the parent of MSE, so using MET's volume here for simplicity.
        if res == 'MSE': res = 'MET'

        # Calculate the SASA/volume.
        percentage = SASA/residue_sizes[res]

        # Add the labels to the mask.
        if percentage <= percentile_33rd: mask[i]=0
        elif percentage > percentile_33rd and percentage <= percentile_66th: mask[i] = 1
        else: mask[i] = 2
    
    return mask, f'{pdb_id}_percentile_mask'

def generate_average_SASA_mask(pdb_id, protein_length, config):
    """
        This function creates the 'average' SASA mask. Average refers to how we categorize low, medium, and high bins.
        In the mask, 0 = low. 1 = medium. 2 = high.
        Low: x<=33rd percentile of SASA/volume.  Medium: 33rd<x<=66th percentile.  High: x>66th percentile
        where x refers to the average between the current and the next residue. 
    """
    # Initialize the mask as the length of the wild type.
    mask = np.zeros(protein_length)

    # Create a dictionary containing volumes of each of the 20 naturally occurring amino acids, found here: https://www.imgt.org/IMGTeducation/Aide-memoire/_UK/aminoacids/abbreviation.html
    residue_sizes = {'ALA':88.6, 'ARG':173.4, 'ASN':114.1, 'ASP':111.1, 'CYS':108.5, 'GLN':143.8, 'GLU':138.4, 'GLY':60.1, 'HIS':153.2, 'ILE':166.7, 'LEU':166.7, 'LYS':168.6, 'MET':162.9, 'PHE': 189.9, 'PRO':112.7, 'SER':89.0, 'THR':116.1, 'TRP':227.8, 'TYR':193.6, 'VAL':140.0}
    
    # Create the path to the saved text file of SASA information per residue, created using PyMol.
    sasa_path = os.path.join(config.get("SASA_path"), f"{pdb_id.lower()}.txt")

    # Sort the text file based on residue position.
    sort_SASA(sasa_path)

    with open(sasa_path, "r") as f:
        lines = f.readlines()
    f.close()

    # Calculate the 33rd and 66th percentiles of the SASA/volumes. 
    percentile_33rd, percentile_66th = calc_SASA_percentiles(lines, residue_sizes)

    for i in range(len(lines)):
        # Find the current residue information.
        tokens_current = lines[i].split()

        # If we don't have a next token (end of sequence), assign 'next' as the current.
        if i == len(lines)-1: 
            tokens_next = lines[i].split()

        # Find the next residue information.
        else: tokens_next = lines[i+1].split()

        # Grab the SASA value at the last token of each line.
        SASA_current = float(tokens_current[-1])
        SASA_next = float(tokens_next[-1])

        # Grab the residue at the 3rd index of each line.
        res_current = tokens_current[3]
        res_next = tokens_next[3]

        # Looks like MET is the parent of MSE, so using MET's volume here for simplicity.
        if res_current == 'MSE': res_current = 'MET' 
        if res_next == 'MSE': res_next = 'MET' 
        
        # Calculate the SASA/volume.
        percentage_current = SASA_current/residue_sizes[res_current]
        percentage_next = SASA_next/residue_sizes[res_next]

        # Add the labels to the mask.
        if i == 0 or i == len(mask)-1:
            if percentage_current <= percentile_33rd: mask[i]=0
            elif percentage_current > percentile_33rd and percentage_current <= percentile_66th: mask[i] = 1
            else: mask[i] = 2

        else:
            # Calculate the average SASA/volume of the current and next residues.
            average = (percentage_current + percentage_next) / 2
            if average <= percentile_33rd: mask[i]=0
            elif average > percentile_33rd and average <= percentile_66th: mask[i] = 1
            else: mask[i] = 2
    
    return mask, f'{pdb_id}_average_mask'

def calc_SASA_percentiles(lines, residue_sizes):
    """
        Helper function to calculate the 33rd and 66th percentiles of the SASA/volume.
    """
    percentages = []

    for line in lines:
        tokens = line.split()

        # Grab the SASA value at the last token of each line.
        SASA = float(tokens[-1])

        # Grab the residue at the 3rd index of each line.
        res = tokens[3]

        # Looks like MET is the parent of MSE, so using MET's volume here for simplicity.
        if res == 'MSE': res = 'MET' 
        percentage = SASA/residue_sizes[res]
        percentages.append(percentage)

    # Calculate percentiles.
    percentile_33rd = np.percentile(percentages, 33)
    percentile_66th = np.percentile(percentages, 66)

    return percentile_33rd, percentile_66th

def extract_line_number(line):
    """
        Helper function for sort_SASA
    """
    return int(line.split()[4][:-1])

def sort_SASA(path):
    """
        Helper function for generating the SASA masks. SASA files from PyMol are not sorted,
        e.g. residue 59, then residue 4, then 16... etc. This function sorts the files by their residue number in the sequence.
    """
    with open(path, "r") as f:
        lines = f.readlines()
    f.close()

    # Sort the lines based on the line number
    sorted_lines = sorted(lines, key=extract_line_number)

    # Write the sorted lines into a new text file
    with open(path, "w") as f:
        f.writelines(sorted_lines)
    f.close()

def save_mask(mask, save_path):
    """
        Save the masks to the provided path as Torch tensors.
    """
    t = torch.from_numpy(mask)
    torch.save(t, save_path)

def main():
    config = load_config("../config/masks.yaml")

    df = pd.read_csv(config.get("csv_path"))
    protein_list = config.get('protein_list')

    if config.get("make_SS"):
        for pdb_id in protein_list:
            protein_length = df.loc[df['pdb_ids'] == pdb_id, 'protein_length'].iloc[0]

            SS_mask, SS_save_name = generate_secondary_structure_mask(pdb_id=pdb_id, protein_length=protein_length, config=config)
            SS_save_path = os.path.join(config.get("save_path"), "SS", f'{SS_save_name}_TESTING.pt')
            save_mask(SS_mask, SS_save_path)
    
    if config.get("make_SASA"):
        for pdb_id in protein_list:
            protein_length = df.loc[df['pdb_ids'] == pdb_id, 'protein_length'].iloc[0]

            if config.get("SASA_type").lower() == "percentile":
                SASA_mask, SASA_save_name = generate_percentile_SASA_mask(pdb_id=pdb_id, protein_length=protein_length, config=config)
            elif config.get("SASA_type").lower() == "average":
                SASA_mask, SASA_save_name = generate_average_SASA_mask(pdb_id=pdb_id, protein_length=protein_length, config=config)
            else:
                SASA_mask, SASA_save_name = generate_original_SASA_mask(pdb_id=pdb_id, protein_length=protein_length, config=config)
            
            SASA_save_path = os.path.join(config.get("save_path"), "SASA", f'{SASA_save_name}_TESTING.pt')
            save_mask(SASA_mask, SASA_save_path)

if __name__ == '__main__':
    main()