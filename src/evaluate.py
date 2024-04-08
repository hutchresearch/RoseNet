"""
RoseNet 2024
Sarah Coffland and Katie Christensen
Evaluation pipeline.
"""

# Standard library imports
import os

# Third party imports
import wandb
import torch
from wandb import Image
from torch.optim import NAdam
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader

# Local imports
from rosenet import RoseNet
from utils import make_one_to_one, load_config
from data.dataset import PDBDataset
from eval_runner import TestRunner, DevRunner

def main(): 
    config = load_config('../config/evaluate.yaml')
    evaluation_pipeline(config)

def evaluation_pipeline(config):
    protein = config.get('protein')

    dictionary = make_dict()
    
    # Initialize Datasets
    dev_set = PDBDataset(config.get('dev_path'))
    test1_set = PDBDataset(config.get('test1_path'))
    test2_set = PDBDataset(config.get('test2_path'))
    test3_set = PDBDataset(config.get('test3_path'))

    # Initialize Dataloaders
    dev_loader = DataLoader(dev_set, batch_size=config.get('batch_size'), shuffle=False, num_workers=config.get('num_workers'), pin_memory=False)
    test1_loader = DataLoader(test1_set, batch_size=config.get('batch_size'), shuffle=False, num_workers=config.get('num_workers'))
    test2_loader = DataLoader(test2_set, batch_size=config.get('batch_size'), shuffle=False, num_workers=config.get('num_workers'))
    test3_loader = DataLoader(test3_set, batch_size=config.get('batch_size'), shuffle=False, num_workers=config.get('num_workers'))

    model = RoseNet(config.get("output_size"), batch_size=config.get("batch_size"), protein_length=config.get("protein_length"), num_blocks=config.get("num_blocks"))
    model.load_state_dict(torch.load(config.get('model_checkpoint_path')))

    optimizer = NAdam(model.parameters(), lr=config.get('lr'))

    if torch.cuda.is_available():
        device = torch.device("cuda") # uses GPU
    else:
        device = torch.device("cpu") # uses CPU

    # Create Runner
    dev_runner = DevRunner(
        dataloader=dev_loader,
        optimizer=optimizer,
        model=model,
        device=device,
    )

    test_runner = TestRunner(
        test1_loader=test1_loader,
        test2_loader=test2_loader,
        test3_loader=test3_loader,
        optimizer=optimizer,
        model=model,
        device=device,
    )

    for epoch in range(config.get('epochs')):
        print("Epoch ", str(epoch), "/", str(config.get('epochs')), "")

        # Validation:
        dev_loss, dev_targets, dev_predictions, dev_inputs = dev_runner.run_epoch(training=False)
        dev_r_2 = r2_score(dev_targets, dev_predictions)
        print(f"\nDev loss: {dev_loss.item()}, dev R^2: {dev_r_2}")
        
        # Calculate and log r2, pearson, one-to-ones.
        eval(dev_targets, dev_predictions, 'dev', config)

        # Check for secondary structure
        helix_indicies, sheet_indicies, no_ss_indicies, all_indicies = check_ss(config, dev_inputs)
        for idx in helix_indicies:
            dictionary['dev_helix']['predictions'].append(dev_predictions[idx])
            dictionary['dev_helix']['targets'].append(dev_targets[idx])
        for idx in sheet_indicies:
            dictionary['dev_sheet']['predictions'].append(dev_predictions[idx])
            dictionary['dev_sheet']['targets'].append(dev_targets[idx])
        for idx in no_ss_indicies:
            dictionary['dev_noss']['predictions'].append(dev_predictions[idx])
            dictionary['dev_noss']['targets'].append(dev_targets[idx])
        for idx in all_indicies:
            dictionary['dev_all']['predictions'].append(dev_predictions[idx])
            dictionary['dev_all']['targets'].append(dev_targets[idx])
        
        # Check for SASA
        low_low_indicies, high_high_indicies, everything_else_indicies = check_sasa(config, dev_inputs)
        for idx in low_low_indicies:
            dictionary['dev_lowlow']['predictions'].append(dev_predictions[idx])
            dictionary['dev_lowlow']['targets'].append(dev_targets[idx])
        for idx in high_high_indicies:
            dictionary['dev_highhigh']['predictions'].append(dev_predictions[idx])
            dictionary['dev_highhigh']['targets'].append(dev_targets[idx])
        for idx in everything_else_indicies:
            dictionary['dev_everythingelse']['predictions'].append(dev_predictions[idx])
            dictionary['dev_everythingelse']['targets'].append(dev_targets[idx])
        
        # Test:
        test1_loss, test1_targets, test1_predictions, test1_inputs = test_runner.run_epoch(1, training=False)
        test1_r_2 = r2_score(test1_targets, test1_predictions)
        print(f"\nTest1 loss: {test1_loss.item()}, test1 R^2: {test1_r_2}")

        # Calculate and log r2, pearson, one-to-ones.
        eval(test1_targets, test1_predictions, 'test1', config)

        # Check for secondary structure
        helix_indicies, sheet_indicies, no_ss_indicies, all_indicies = check_ss(config, test1_inputs)
        for idx in helix_indicies:
            dictionary['test1_helix']['predictions'].append(test1_predictions[idx])
            dictionary['test1_helix']['targets'].append(test1_targets[idx])
        for idx in sheet_indicies:
            dictionary['test1_sheet']['predictions'].append(test1_predictions[idx])
            dictionary['test1_sheet']['targets'].append(test1_targets[idx])
        for idx in no_ss_indicies:
            dictionary['test1_noss']['predictions'].append(test1_predictions[idx])
            dictionary['test1_noss']['targets'].append(test1_targets[idx])
        for idx in all_indicies:
            dictionary['test1_all']['predictions'].append(test1_predictions[idx])
            dictionary['test1_all']['targets'].append(test1_targets[idx])
        
        # Check for SASA
        low_low_indicies, high_high_indicies, everything_else_indicies = check_sasa(config, test1_inputs)
        for idx in low_low_indicies:
            dictionary['test1_lowlow']['predictions'].append(test1_predictions[idx])
            dictionary['test1_lowlow']['targets'].append(test1_targets[idx])
        for idx in high_high_indicies:
            dictionary['test1_highhigh']['predictions'].append(test1_predictions[idx])
            dictionary['test1_highhigh']['targets'].append(test1_targets[idx])
        for idx in everything_else_indicies:
            dictionary['test1_everythingelse']['predictions'].append(test1_predictions[idx])
            dictionary['test1_everythingelse']['targets'].append(test1_targets[idx])
        
        test2_loss, test2_targets, test2_predictions, _ = test_runner.run_epoch(2, training=False)
        test2_r_2 = r2_score(test2_targets, test2_predictions)
        print(f"\nTest2 loss: {test2_loss.item()}, test2 R^2: {test2_r_2}")

        # Calculate and log r2, pearson, one-to-ones.
        eval(test2_targets, test2_predictions, 'test2', config)
        
        test3_loss, test3_targets, test3_predictions, _ = test_runner.run_epoch(3, training=False)
        test3_r_2 = r2_score(test3_targets, test3_predictions)
        print(f"\nTest3 loss: {test3_loss.item()}, test3 R^2: {test3_r_2}")

        # Calculate and log r2, pearson, one-to-ones.
        eval(test3_targets, test3_predictions, 'test3', config)

    dev_SS_list = [dictionary['dev_helix']['predictions'], dictionary['dev_helix']['targets'], dictionary['dev_sheet']['predictions'], dictionary['dev_sheet']['targets'], dictionary['dev_all']['predictions'], dictionary['dev_all']['targets'], dictionary['dev_noss']['predictions'], dictionary['dev_noss']['targets']]
    dev_SASA_list = [dictionary['dev_lowlow']['predictions'], dictionary['dev_lowlow']['targets'], dictionary['dev_highhigh']['predictions'], dictionary['dev_highhigh']['targets'], dictionary['dev_everythingelse']['predictions'], dictionary['dev_everythingelse']['targets']]    
    make_ss_sasa_figure(config, dev_SS_list, dev_SASA_list, protein, 'Validation')
    
    test_SS_list = [dictionary['test1_helix']['predictions'], dictionary['test1_helix']['targets'], dictionary['test1_sheet']['predictions'], dictionary['test1_sheet']['targets'], dictionary['test1_all']['predictions'], dictionary['test1_all']['targets'], dictionary['test1_noss']['predictions'], dictionary['test1_noss']['targets']]
    test_SASA_list = [dictionary['test1_lowlow']['predictions'], dictionary['test1_lowlow']['targets'], dictionary['test1_highhigh']['predictions'], dictionary['test1_highhigh']['targets'], dictionary['test1_everythingelse']['predictions'], dictionary['test1_everythingelse']['targets']]   
    make_ss_sasa_figure(config, test_SS_list, test_SASA_list, protein, 'Test')

def make_dict():
    # Initialize a dictionary to hold lists for each category
    dictionary = {
        "dev_all": {"predictions": [], 'targets': []},
        "dev_noss": {"predictions": [], 'targets': []},
        "dev_helix": {"predictions": [], 'targets': []},
        "dev_sheet": {"predictions": [], 'targets': []},
        "dev_lowlow": {"predictions": [], 'targets': []},
        "dev_highhigh": {"predictions": [], 'targets': []},
        "dev_everythingelse": {"predictions": [], 'targets': []},
        "test1_all": {"predictions": [], 'targets': []},
        "test1_noss": {"predictions": [], 'targets': []},
        "test1_helix": {"predictions": [], 'targets': []},
        "test1_sheet": {"predictions": [], 'targets': []},
        "test1_lowlow": {"predictions": [], 'targets': []},
        "test1_highhigh": {"predictions": [], 'targets': []},
        "test1_everythingelse": {"predictions": [], 'targets': []}
    }
    return dictionary

def eval(targets, predictions, type, config):
    """
        Calculate the R^2 and Pearson correlation coefficients of the validation predictions for each Rosetta score and make one to one plots.
    """
    for i, score in enumerate(config.get('scores')):
        rosetta_score_r2 = r2_score(targets[:,i], predictions[:,i])
        pearson_correlation, p_value = pearsonr(targets[:,i], predictions[:,i])
        
        print(f"{type}: {score}_r2: {rosetta_score_r2}, {score}_pearson_correlation: {pearson_correlation}, p_value: {p_value} \n")

        protein = config.get("protein")
        make_one_to_one(config, targets[:,i], predictions[:,i], f"{protein}_{score}_one_to_one")

def check_sasa(config, inputs):
    mask = torch.load(config.get('SASA_mask_path'))

    low_low_indicies = []
    high_high_indicies = []
    everything_else_indicies = []

    if config.get('SASA_type2').lower() == 'combined':
        for i, input in enumerate(inputs):
            if config.get('SASA_type').lower() == 'average':
                pos1 = input[0].item()-1 
                pos2 = input[2].item()-2 
            elif config.get('SASA_type').lower() == 'before':
                #CARE ABOUT THE ONE BEFORE:
                pos1 = input[0].item()-2
                pos2 = input[2].item()-3 
            elif config.get('SASA_type').lower() == 'after': 
                #CARE ABOUT THE ONE AFTER:
                pos1 = input[0].item()-1
                pos2 = input[2].item()-2

            if pos1 >= len(mask): pos1 = len(mask)-1
            if pos2 >= len(mask): pos2 = len(mask)-1

            # Both low
            if mask[pos1] == 0 and mask[pos2] == 0: low_low_indicies.append(i)
            # One low, one medium
            elif mask[pos1] == 0 and mask[pos2] == 1: low_low_indicies.append(i)
            # One low, one medium
            elif mask[pos1] == 1 and mask[pos2] == 0: low_low_indicies.append(i)
            # Both high
            elif mask[pos1] == 2 and mask[pos2] == 2: high_high_indicies.append(i)
            # One high, one medium
            elif mask[pos1] == 1 and mask[pos2] == 2: high_high_indicies.append(i)
            # One high, one medium
            elif mask[pos1] == 2 and mask[pos2] == 1: high_high_indicies.append(i)
            # Everything else
            else: everything_else_indicies.append(i)
        # breakpoint()

    else:
        for i, input in enumerate(inputs):
            if config.get('SASA_type').lower() == 'average':
                pos1 = input[0].item()-1 
                pos2 = input[2].item()-2 
            elif config.get('SASA_type').lower() == 'before':
                #CARE ABOUT THE ONE BEFORE:
                pos1 = input[0].item()-2
                pos2 = input[2].item()-3 
            elif config.get('SASA_type').lower() == 'after': 
                #CARE ABOUT THE ONE AFTER:
                pos1 = input[0].item()-1
                pos2 = input[2].item()-2

            if pos1 >= len(mask): pos1 = len(mask)-1
            if pos2 >= len(mask): pos2 = len(mask)-1

            # Both low
            if mask[pos1] == 0 and mask[pos2] == 0: low_low_indicies.append(i)
            # Both high
            elif mask[pos1] == 2 and mask[pos2] == 2: high_high_indicies.append(i)
            # Everything else
            else: everything_else_indicies.append(i)
    
    return low_low_indicies, high_high_indicies, everything_else_indicies

def check_ss(config, inputs):
    mask = torch.load(config.get('SS_mask_path'))

    helix_indicies = []
    sheet_indicies = []
    all_indicies = []
    no_ss_indicies = []

    for i, input in enumerate(inputs):
        pos1_before = input[0].item()-2
        pos1_after = input[0].item()-1

        pos2_before = input[2].item()-3
        pos2_after = input[2].item()-2

        if pos1_before >= len(mask) or pos1_after >= len(mask): pos1_after = len(mask)-1
        if pos2_before >= len(mask) or pos2_after >= len(mask): pos2_after = len(mask)-1

        ss_inserted = False

        if mask[pos1_before] == 1 and mask[pos1_after] == 1: 
            ss_inserted = True
            helix_indicies.append(i)
        elif mask[pos2_before] == 1 and mask[pos2_after] == 1: 
            ss_inserted = True
            helix_indicies.append(i)

        if mask[pos1_before] == 2 and mask[pos1_after] == 2: 
            ss_inserted = True
            sheet_indicies.append(i)
        elif mask[pos2_before] == 2 and mask[pos2_after] == 2: 
            ss_inserted = True
            sheet_indicies.append(i)

        if not ss_inserted: no_ss_indicies.append(i)

        all_indicies.append(i)

    return helix_indicies, sheet_indicies, no_ss_indicies, all_indicies

def make_SASA_mask_figure(config, list, ax=None):
    scores = config.get('scores')
    
    lowlow_predictions, lowlow_targets, highhigh_predictions, highhigh_targets, ee_predictions, ee_targets = list

    lowlow = False 
    highhigh = False 
    ee = False

    if lowlow_predictions is not None and len(lowlow_predictions) > 0:
        lowlow = True

        lowlow_predictions = torch.vstack(lowlow_predictions)
        lowlow_targets = torch.vstack(lowlow_targets)

        # Calculate pearson for each rosetta score.
        ll_fa_atr_pearson_correlation, _ = pearsonr(        lowlow_targets[:,0],  lowlow_predictions[:,0])
        ll_fa_rep_pearson_correlation, _ = pearsonr(        lowlow_targets[:,1],  lowlow_predictions[:,1])
        ll_fa_sol_pearson_correlation, _ = pearsonr(        lowlow_targets[:,2],  lowlow_predictions[:,2])
        ll_fa_intra_rep_pearson_correlation, _ = pearsonr(  lowlow_targets[:,3],  lowlow_predictions[:,3])
        ll_fa_intra_sol_pearson_correlation, _ = pearsonr(  lowlow_targets[:,4],  lowlow_predictions[:,4])
        ll_lk_ball_wtd_pearson_correlation, _ = pearsonr(   lowlow_targets[:,5],  lowlow_predictions[:,5])
        ll_fa_elec_pearson_correlation, _ = pearsonr(       lowlow_targets[:,6],  lowlow_predictions[:,6])
        ll_pro_close_pearson_correlation, _ = pearsonr(     lowlow_targets[:,7],  lowlow_predictions[:,7])
        ll_hbond_sr_bb_pearson_correlation, _ = pearsonr(   lowlow_targets[:,8],  lowlow_predictions[:,8])
        ll_hbond_lr_bb_pearson_correlation, _ = pearsonr(   lowlow_targets[:,9],  lowlow_predictions[:,9])
        ll_hbond_bb_sc_pearson_correlation, _ = pearsonr(   lowlow_targets[:,10], lowlow_predictions[:,10])
        ll_hbond_sc_pearson_correlation, _ = pearsonr(      lowlow_targets[:,11], lowlow_predictions[:,11])
        ll_omega_pearson_correlation, _ = pearsonr(         lowlow_targets[:,12], lowlow_predictions[:,12])
        ll_fa_dun_pearson_correlation, _ = pearsonr(        lowlow_targets[:,13], lowlow_predictions[:,13])
        ll_p_aa_pp_pearson_correlation, _ = pearsonr(       lowlow_targets[:,14], lowlow_predictions[:,14])
        ll_ref_pearson_correlation, _ = pearsonr(           lowlow_targets[:,15], lowlow_predictions[:,15])
        ll_rama_prepro_pearson_correlation, _ = pearsonr(   lowlow_targets[:,16], lowlow_predictions[:,16])
        ll_total_pearson_correlation, _ = pearsonr(         lowlow_targets[:,17], lowlow_predictions[:,17])

    if highhigh_predictions is not None and len(highhigh_predictions) > 0:
        highhigh = True

        highhigh_predictions = torch.vstack(highhigh_predictions)
        highhigh_targets = torch.vstack(highhigh_targets)
        
        # Calculate pearson for each rosetta score for the sheet ss.
        hh_fa_atr_pearson_correlation, _ = pearsonr(        highhigh_targets[:,0],  highhigh_predictions[:,0])
        hh_fa_rep_pearson_correlation, _ = pearsonr(        highhigh_targets[:,1],  highhigh_predictions[:,1])
        hh_fa_sol_pearson_correlation, _ = pearsonr(        highhigh_targets[:,2],  highhigh_predictions[:,2])
        hh_fa_intra_rep_pearson_correlation, _ = pearsonr(  highhigh_targets[:,3],  highhigh_predictions[:,3])
        hh_fa_intra_sol_pearson_correlation, _ = pearsonr(  highhigh_targets[:,4],  highhigh_predictions[:,4])
        hh_lk_ball_wtd_pearson_correlation, _ = pearsonr(   highhigh_targets[:,5],  highhigh_predictions[:,5])
        hh_fa_elec_pearson_correlation, _ = pearsonr(       highhigh_targets[:,6],  highhigh_predictions[:,6])
        hh_pro_close_pearson_correlation, _ = pearsonr(     highhigh_targets[:,7],  highhigh_predictions[:,7])
        hh_hbond_sr_bb_pearson_correlation, _ = pearsonr(   highhigh_targets[:,8],  highhigh_predictions[:,8])
        hh_hbond_lr_bb_pearson_correlation, _ = pearsonr(   highhigh_targets[:,9],  highhigh_predictions[:,9])
        hh_hbond_bb_sc_pearson_correlation, _ = pearsonr(   highhigh_targets[:,10], highhigh_predictions[:,10])
        hh_hbond_sc_pearson_correlation, _ = pearsonr(      highhigh_targets[:,11], highhigh_predictions[:,11])
        hh_omega_pearson_correlation, _ = pearsonr(         highhigh_targets[:,12], highhigh_predictions[:,12])
        hh_fa_dun_pearson_correlation, _ = pearsonr(        highhigh_targets[:,13], highhigh_predictions[:,13])
        hh_p_aa_pp_pearson_correlation, _ = pearsonr(       highhigh_targets[:,14], highhigh_predictions[:,14])
        hh_ref_pearson_correlation, _ = pearsonr(           highhigh_targets[:,15], highhigh_predictions[:,15])
        hh_rama_prepro_pearson_correlation, _ = pearsonr(   highhigh_targets[:,16], highhigh_predictions[:,16])
        hh_total_pearson_correlation, _ = pearsonr(         highhigh_targets[:,17], highhigh_predictions[:,17])

    if ee_predictions is not None and len(ee_predictions) > 0:
        ee = True

        ee_predictions = torch.vstack(ee_predictions)
        ee_targets = torch.vstack(ee_targets)
        
        # Calculate pearson for each rosetta score for the sheet ss.
        ee_fa_atr_pearson_correlation, _ = pearsonr(        ee_targets[:,0],  ee_predictions[:,0])
        ee_fa_rep_pearson_correlation, _ = pearsonr(        ee_targets[:,1],  ee_predictions[:,1])
        ee_fa_sol_pearson_correlation, _ = pearsonr(        ee_targets[:,2],  ee_predictions[:,2])
        ee_fa_intra_rep_pearson_correlation, _ = pearsonr(  ee_targets[:,3],  ee_predictions[:,3])
        ee_fa_intra_sol_pearson_correlation, _ = pearsonr(  ee_targets[:,4],  ee_predictions[:,4])
        ee_lk_ball_wtd_pearson_correlation, _ = pearsonr(   ee_targets[:,5],  ee_predictions[:,5])
        ee_fa_elec_pearson_correlation, _ = pearsonr(       ee_targets[:,6],  ee_predictions[:,6])
        ee_pro_close_pearson_correlation, _ = pearsonr(     ee_targets[:,7],  ee_predictions[:,7])
        ee_hbond_sr_bb_pearson_correlation, _ = pearsonr(   ee_targets[:,8],  ee_predictions[:,8])
        ee_hbond_lr_bb_pearson_correlation, _ = pearsonr(   ee_targets[:,9],  ee_predictions[:,9])
        ee_hbond_bb_sc_pearson_correlation, _ = pearsonr(   ee_targets[:,10], ee_predictions[:,10])
        ee_hbond_sc_pearson_correlation, _ = pearsonr(      ee_targets[:,11], ee_predictions[:,11])
        ee_omega_pearson_correlation, _ = pearsonr(         ee_targets[:,12], ee_predictions[:,12])
        ee_fa_dun_pearson_correlation, _ = pearsonr(        ee_targets[:,13], ee_predictions[:,13])
        ee_p_aa_pp_pearson_correlation, _ = pearsonr(       ee_targets[:,14], ee_predictions[:,14])
        ee_ref_pearson_correlation, _ = pearsonr(           ee_targets[:,15], ee_predictions[:,15])
        ee_rama_prepro_pearson_correlation, _ = pearsonr(   ee_targets[:,16], ee_predictions[:,16])
        ee_total_pearson_correlation, _ = pearsonr(         ee_targets[:,17], ee_predictions[:,17])
    
    ax = ax or plt.gca()

    if highhigh:
        line, = ax.plot(scores, [hh_fa_atr_pearson_correlation, hh_fa_rep_pearson_correlation, hh_fa_sol_pearson_correlation, hh_fa_intra_rep_pearson_correlation, hh_fa_intra_sol_pearson_correlation,
                            hh_lk_ball_wtd_pearson_correlation, hh_fa_elec_pearson_correlation, hh_pro_close_pearson_correlation, hh_hbond_sr_bb_pearson_correlation, hh_hbond_lr_bb_pearson_correlation,
                            hh_hbond_bb_sc_pearson_correlation, hh_hbond_sc_pearson_correlation, hh_omega_pearson_correlation, hh_fa_dun_pearson_correlation, hh_p_aa_pp_pearson_correlation,
                            hh_ref_pearson_correlation, hh_rama_prepro_pearson_correlation, hh_total_pearson_correlation], marker='s', linestyle='-', label='high high', color='#ff7f0e')
    
    if lowlow:
        line, = ax.plot(scores, [ll_fa_atr_pearson_correlation, ll_fa_rep_pearson_correlation, ll_fa_sol_pearson_correlation, ll_fa_intra_rep_pearson_correlation, ll_fa_intra_sol_pearson_correlation,
                            ll_lk_ball_wtd_pearson_correlation, ll_fa_elec_pearson_correlation, ll_pro_close_pearson_correlation, ll_hbond_sr_bb_pearson_correlation, ll_hbond_lr_bb_pearson_correlation,
                            ll_hbond_bb_sc_pearson_correlation, ll_hbond_sc_pearson_correlation, ll_omega_pearson_correlation, ll_fa_dun_pearson_correlation, ll_p_aa_pp_pearson_correlation,
                            ll_ref_pearson_correlation, ll_rama_prepro_pearson_correlation, ll_total_pearson_correlation], marker='o', linestyle='-', label='low low', color='#1f77b4')
    if ee:
        line, = ax.plot(scores, [ee_fa_atr_pearson_correlation, ee_fa_rep_pearson_correlation, ee_fa_sol_pearson_correlation, ee_fa_intra_rep_pearson_correlation, ee_fa_intra_sol_pearson_correlation,
                            ee_lk_ball_wtd_pearson_correlation, ee_fa_elec_pearson_correlation, ee_pro_close_pearson_correlation, ee_hbond_sr_bb_pearson_correlation, ee_hbond_lr_bb_pearson_correlation,
                            ee_hbond_bb_sc_pearson_correlation, ee_hbond_sc_pearson_correlation, ee_omega_pearson_correlation, ee_fa_dun_pearson_correlation, ee_p_aa_pp_pearson_correlation,
                            ee_ref_pearson_correlation, ee_rama_prepro_pearson_correlation, ee_total_pearson_correlation], marker='^', linestyle='-', label='everything else', color='#2ca02c')
    if lowlow or highhigh or ee:
        ax.set_xlabel('Rosetta score', fontsize=14)
        ax.set_xticks(range(len(scores)))
        ax.set_xticklabels(scores, rotation=45, ha='right', fontsize=12)
        ax.tick_params(axis='y', labelsize=12)
        ax.set_ylabel('Pearson Correlation Coefficient', fontsize=14)
        ax.legend(loc='upper left', bbox_to_anchor=(1.005, 1), borderaxespad=0., fontsize=12)

        return line

def make_SS_mask_figure(config, list, protein, type, ax=None):
    scores = config.get('scores')

    h_predictions, h_targets, s_predictions, s_targets, all_predictions, all_targets, no_predictions, no_targets = list

    helix = False 
    sheet = False 
    all = False
    none = False

    if h_predictions is not None and len(h_predictions) > 0:
        helix = True

        h_predictions = torch.vstack(h_predictions)
        h_targets = torch.vstack(h_targets)

        # Calculate pearson for each rosetta score for the helix ss.
        h_fa_atr_pearson_correlation, _ = pearsonr(        h_targets[:,0],  h_predictions[:,0])
        h_fa_rep_pearson_correlation, _ = pearsonr(        h_targets[:,1],  h_predictions[:,1])
        h_fa_sol_pearson_correlation, _ = pearsonr(        h_targets[:,2],  h_predictions[:,2])
        h_fa_intra_rep_pearson_correlation, _ = pearsonr(  h_targets[:,3],  h_predictions[:,3])
        h_fa_intra_sol_pearson_correlation, _ = pearsonr(  h_targets[:,4],  h_predictions[:,4])
        h_lk_ball_wtd_pearson_correlation, _ = pearsonr(   h_targets[:,5],  h_predictions[:,5])
        h_fa_elec_pearson_correlation, _ = pearsonr(       h_targets[:,6],  h_predictions[:,6])
        h_pro_close_pearson_correlation, _ = pearsonr(     h_targets[:,7],  h_predictions[:,7])
        h_hbond_sr_bb_pearson_correlation, _ = pearsonr(   h_targets[:,8],  h_predictions[:,8])
        h_hbond_lr_bb_pearson_correlation, _ = pearsonr(   h_targets[:,9],  h_predictions[:,9])
        h_hbond_bb_sc_pearson_correlation, _ = pearsonr(   h_targets[:,10], h_predictions[:,10])
        h_hbond_sc_pearson_correlation, _ = pearsonr(      h_targets[:,11], h_predictions[:,11])
        h_omega_pearson_correlation, _ = pearsonr(         h_targets[:,12], h_predictions[:,12])
        h_fa_dun_pearson_correlation, _ = pearsonr(        h_targets[:,13], h_predictions[:,13])
        h_p_aa_pp_pearson_correlation, _ = pearsonr(       h_targets[:,14], h_predictions[:,14])
        h_ref_pearson_correlation, _ = pearsonr(           h_targets[:,15], h_predictions[:,15])
        h_rama_prepro_pearson_correlation, _ = pearsonr(   h_targets[:,16], h_predictions[:,16])
        h_total_pearson_correlation, _ = pearsonr(         h_targets[:,17], h_predictions[:,17])

    if s_predictions is not None and len(s_predictions) > 0:
        sheet = True

        s_predictions = torch.vstack(s_predictions)
        s_targets = torch.vstack(s_targets)
        
        # Calculate pearson for each rosetta score for the sheet ss.
        s_fa_atr_pearson_correlation, _ = pearsonr(        s_targets[:,0],  s_predictions[:,0])
        s_fa_rep_pearson_correlation, _ = pearsonr(        s_targets[:,1],  s_predictions[:,1])
        s_fa_sol_pearson_correlation, _ = pearsonr(        s_targets[:,2],  s_predictions[:,2])
        s_fa_intra_rep_pearson_correlation, _ = pearsonr(  s_targets[:,3],  s_predictions[:,3])
        s_fa_intra_sol_pearson_correlation, _ = pearsonr(  s_targets[:,4],  s_predictions[:,4])
        s_lk_ball_wtd_pearson_correlation, _ = pearsonr(   s_targets[:,5],  s_predictions[:,5])
        s_fa_elec_pearson_correlation, _ = pearsonr(       s_targets[:,6],  s_predictions[:,6])
        s_pro_close_pearson_correlation, _ = pearsonr(     s_targets[:,7],  s_predictions[:,7])
        s_hbond_sr_bb_pearson_correlation, _ = pearsonr(   s_targets[:,8],  s_predictions[:,8])
        s_hbond_lr_bb_pearson_correlation, _ = pearsonr(   s_targets[:,9],  s_predictions[:,9])
        s_hbond_bb_sc_pearson_correlation, _ = pearsonr(   s_targets[:,10], s_predictions[:,10])
        s_hbond_sc_pearson_correlation, _ = pearsonr(      s_targets[:,11], s_predictions[:,11])
        s_omega_pearson_correlation, _ = pearsonr(         s_targets[:,12], s_predictions[:,12])
        s_fa_dun_pearson_correlation, _ = pearsonr(        s_targets[:,13], s_predictions[:,13])
        s_p_aa_pp_pearson_correlation, _ = pearsonr(       s_targets[:,14], s_predictions[:,14])
        s_ref_pearson_correlation, _ = pearsonr(           s_targets[:,15], s_predictions[:,15])
        s_rama_prepro_pearson_correlation, _ = pearsonr(   s_targets[:,16], s_predictions[:,16])
        s_total_pearson_correlation, _ = pearsonr(         s_targets[:,17], s_predictions[:,17])
    
    if all_predictions is not None and len(all_predictions) > 0:
        all = True

        all_predictions = torch.vstack(all_predictions)
        all_targets = torch.vstack(all_targets)
        
        # Calculate pearson for each rosetta score for the sheet ss.
        all_fa_atr_pearson_correlation, _ = pearsonr(        all_targets[:,0],  all_predictions[:,0])
        all_fa_rep_pearson_correlation, _ = pearsonr(        all_targets[:,1],  all_predictions[:,1])
        all_fa_sol_pearson_correlation, _ = pearsonr(        all_targets[:,2],  all_predictions[:,2])
        all_fa_intra_rep_pearson_correlation, _ = pearsonr(  all_targets[:,3],  all_predictions[:,3])
        all_fa_intra_sol_pearson_correlation, _ = pearsonr(  all_targets[:,4],  all_predictions[:,4])
        all_lk_ball_wtd_pearson_correlation, _ = pearsonr(   all_targets[:,5],  all_predictions[:,5])
        all_fa_elec_pearson_correlation, _ = pearsonr(       all_targets[:,6],  all_predictions[:,6])
        all_pro_close_pearson_correlation, _ = pearsonr(     all_targets[:,7],  all_predictions[:,7])
        all_hbond_sr_bb_pearson_correlation, _ = pearsonr(   all_targets[:,8],  all_predictions[:,8])
        all_hbond_lr_bb_pearson_correlation, _ = pearsonr(   all_targets[:,9],  all_predictions[:,9])
        all_hbond_bb_sc_pearson_correlation, _ = pearsonr(   all_targets[:,10], all_predictions[:,10])
        all_hbond_sc_pearson_correlation, _ = pearsonr(      all_targets[:,11], all_predictions[:,11])
        all_omega_pearson_correlation, _ = pearsonr(         all_targets[:,12], all_predictions[:,12])
        all_fa_dun_pearson_correlation, _ = pearsonr(        all_targets[:,13], all_predictions[:,13])
        all_p_aa_pp_pearson_correlation, _ = pearsonr(       all_targets[:,14], all_predictions[:,14])
        all_ref_pearson_correlation, _ = pearsonr(           all_targets[:,15], all_predictions[:,15])
        all_rama_prepro_pearson_correlation, _ = pearsonr(   all_targets[:,16], all_predictions[:,16])
        all_total_pearson_correlation, _ = pearsonr(         all_targets[:,17], all_predictions[:,17])

    if no_predictions is not None and len(no_predictions) > 0:
        none = True

        no_predictions = torch.vstack(no_predictions)
        no_targets = torch.vstack(no_targets)
        
        # Calculate pearson for each rosetta score for the sheet ss.
        no_fa_atr_pearson_correlation, _ = pearsonr(        no_targets[:,0],  no_predictions[:,0])
        no_fa_rep_pearson_correlation, _ = pearsonr(        no_targets[:,1],  no_predictions[:,1])
        no_fa_sol_pearson_correlation, _ = pearsonr(        no_targets[:,2],  no_predictions[:,2])
        no_fa_intra_rep_pearson_correlation, _ = pearsonr(  no_targets[:,3],  no_predictions[:,3])
        no_fa_intra_sol_pearson_correlation, _ = pearsonr(  no_targets[:,4],  no_predictions[:,4])
        no_lk_ball_wtd_pearson_correlation, _ = pearsonr(   no_targets[:,5],  no_predictions[:,5])
        no_fa_elec_pearson_correlation, _ = pearsonr(       no_targets[:,6],  no_predictions[:,6])
        no_pro_close_pearson_correlation, _ = pearsonr(     no_targets[:,7],  no_predictions[:,7])
        no_hbond_sr_bb_pearson_correlation, _ = pearsonr(   no_targets[:,8],  no_predictions[:,8])
        no_hbond_lr_bb_pearson_correlation, _ = pearsonr(   no_targets[:,9],  no_predictions[:,9])
        no_hbond_bb_sc_pearson_correlation, _ = pearsonr(   no_targets[:,10], no_predictions[:,10])
        no_hbond_sc_pearson_correlation, _ = pearsonr(      no_targets[:,11], no_predictions[:,11])
        no_omega_pearson_correlation, _ = pearsonr(         no_targets[:,12], no_predictions[:,12])
        no_fa_dun_pearson_correlation, _ = pearsonr(        no_targets[:,13], no_predictions[:,13])
        no_p_aa_pp_pearson_correlation, _ = pearsonr(       no_targets[:,14], no_predictions[:,14])
        no_ref_pearson_correlation, _ = pearsonr(           no_targets[:,15], no_predictions[:,15])
        no_rama_prepro_pearson_correlation, _ = pearsonr(   no_targets[:,16], no_predictions[:,16])
        no_total_pearson_correlation, _ = pearsonr(         no_targets[:,17], no_predictions[:,17])
    
    ax = ax or plt.gca()

    if none:
        line, = ax.plot(scores, [no_fa_atr_pearson_correlation, no_fa_rep_pearson_correlation, no_fa_sol_pearson_correlation, no_fa_intra_rep_pearson_correlation, no_fa_intra_sol_pearson_correlation,
                            no_lk_ball_wtd_pearson_correlation, no_fa_elec_pearson_correlation, no_pro_close_pearson_correlation, no_hbond_sr_bb_pearson_correlation, no_hbond_lr_bb_pearson_correlation,
                            no_hbond_bb_sc_pearson_correlation, no_hbond_sc_pearson_correlation, no_omega_pearson_correlation, no_fa_dun_pearson_correlation, no_p_aa_pp_pearson_correlation,
                            no_ref_pearson_correlation, no_rama_prepro_pearson_correlation, no_total_pearson_correlation], marker='d', linestyle='-', label='no SS', color='#ffdc00') #yellow
        
    if helix:
        line, = ax.plot(scores, [h_fa_atr_pearson_correlation, h_fa_rep_pearson_correlation, h_fa_sol_pearson_correlation, h_fa_intra_rep_pearson_correlation, h_fa_intra_sol_pearson_correlation,
                            h_lk_ball_wtd_pearson_correlation, h_fa_elec_pearson_correlation, h_pro_close_pearson_correlation, h_hbond_sr_bb_pearson_correlation, h_hbond_lr_bb_pearson_correlation,
                            h_hbond_bb_sc_pearson_correlation, h_hbond_sc_pearson_correlation, h_omega_pearson_correlation, h_fa_dun_pearson_correlation, h_p_aa_pp_pearson_correlation,
                            h_ref_pearson_correlation, h_rama_prepro_pearson_correlation, h_total_pearson_correlation], marker='s', linestyle='-', label='alpha helix', color='#ff7f0e') #orange
    if sheet:
        line, = ax.plot(scores, [s_fa_atr_pearson_correlation, s_fa_rep_pearson_correlation, s_fa_sol_pearson_correlation, s_fa_intra_rep_pearson_correlation, s_fa_intra_sol_pearson_correlation,
                            s_lk_ball_wtd_pearson_correlation, s_fa_elec_pearson_correlation, s_pro_close_pearson_correlation, s_hbond_sr_bb_pearson_correlation, s_hbond_lr_bb_pearson_correlation,
                            s_hbond_bb_sc_pearson_correlation, s_hbond_sc_pearson_correlation, s_omega_pearson_correlation, s_fa_dun_pearson_correlation, s_p_aa_pp_pearson_correlation,
                            s_ref_pearson_correlation, s_rama_prepro_pearson_correlation, s_total_pearson_correlation], marker='o', linestyle='-', label='beta sheet', color='#1f77b4')   #blue
    if all:
        line, = ax.plot(scores, [all_fa_atr_pearson_correlation, all_fa_rep_pearson_correlation, all_fa_sol_pearson_correlation, all_fa_intra_rep_pearson_correlation, all_fa_intra_sol_pearson_correlation,
                            all_lk_ball_wtd_pearson_correlation, all_fa_elec_pearson_correlation, all_pro_close_pearson_correlation, all_hbond_sr_bb_pearson_correlation, all_hbond_lr_bb_pearson_correlation,
                            all_hbond_bb_sc_pearson_correlation, all_hbond_sc_pearson_correlation, all_omega_pearson_correlation, all_fa_dun_pearson_correlation, all_p_aa_pp_pearson_correlation,
                            all_ref_pearson_correlation, all_rama_prepro_pearson_correlation, all_total_pearson_correlation], marker='^', linestyle='-', label='all data', color='#2ca02c') #green
    
    if helix or sheet or all or none:
        ax.set_title(protein + " - " + type, fontsize=16)
        ax.set_xticklabels([])
        ax.tick_params(axis='y', labelsize=12)
        ax.set_ylabel('Pearson Correlation Coefficient', fontsize=14)

        # Changing the order of the legend
        handles, labels = ax.get_legend_handles_labels()
        if helix and sheet and all and none:
            order = [1,2,3,0]  # Specify the order of legend items
        elif none==False:
            order = [0,1,2]
        else:
            order = [1,2,0]
        ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='upper left', bbox_to_anchor=(1.005, 1), borderaxespad=0., fontsize=12)

        return line

def make_ss_sasa_figure(config, ss_list, sasa_list, protein, type):
    fig, (ax1,ax2) = plt.subplots(nrows=2, figsize=(9,10))
    
    make_SS_mask_figure(config, ss_list, protein, type, ax1)
    make_SASA_mask_figure(config, sasa_list, ax2)

    plt.tight_layout()

    savepath = os.path.join(config.get('SASA_SS_figure_save_path'), f'{protein}_{type}')
    plt.savefig(savepath)

if __name__ == "__main__":
    main()