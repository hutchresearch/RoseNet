# RoseNet
Sarah Coffland, Katie Christensen  |  Western Washington University, Department of Computer Science, 2024

Studying the structural and functional implications of protein mutations is an important task in computational biology and bioinformatics. We predict energy metrics of proteins with double amino acid insertions or deletions (InDels). We train three of these models on our data sets containing the exhaustive double InDel mutations for three proteins, as well as another three proteins for which roughly 145k random mutants with two InDels have been generated. Our models make use of the identities of the residues, and the positions at which these residues are inserted.

## Data
* All train/validation/test1/test2/test3 splits for each protein live in the data splits directory.
* To create new data splits
    * Update the configuration file data.yaml under the config/ directory
    * Run the script to create the data splits 
        ```
        cd src/
        python3 data_split.py
        ``` 

## Training
* Update the configuration file main.yaml under the config/ directory
* Run the training pipeline
    ```
       cd src/
       python3 pipeline.py
    ```

## Evaluating
* Create the solvation accessible surface area (SASA) and secondary structure (SS) masks.
    * Update the configuration file masks.yaml under the config/ directory
    * Run the script to create the masks 
        ```
        cd src/
        python3 make_masks.py
        ``` 
* Update the configuration file evaluate.yaml under the config/ directory
* Run the evaluation pipeline
    ```
       cd src/
       python3 evaluate.py
    ```

### Acknowledgements
Dr. Brian Hutchinson (Western Washington University) and Dr. Filip Jagodzinski (Western Washington University)

### Rosetta Scores and Definitions
| Rosetta Score  | Definition |
| ------------- | ------------- |
| fa_atr  | Lennard-Jones attractive between atoms in different residues  |
| fa_rep  | Lennard-Jones repulsive between atoms in different residues  |
| fa_sol  | Lazaridis-Karplus solvation energy  |
| fa_intra_sol_xover4  | Intra-residue Lazaridis-Karplus solvation energy  |
| lk_ball_wtd  | Asymmetric solvation energy  |
| fa_intra_rep  | Lennard-Jones repulsive between atoms in the same residue  |
| fa_elec  | Coulombic electrostatic potential with a distance-dependent dielectric  |
| pro_close  | Proline ring closure energy and energy of psi angle of preceding residue  |
| hbond_sr_bb  | Backbone-backbone hbonds close in primary sequence  |
| hbond_lr_bb  | Backbone-backbone hbonds distant in primary sequence  |
| hbond_bb_sc   | Sidechain-backbone hydrogen bond energy  |
| hbond_sc  | Sidechain-sidechain hydrogen bond energy  |
| dslf_fa13  | Disulfide geometry potential  |
| rama_prepro  | Ramachandran preferences (with separate lookup tables for pre-proline positions and other positions)  |
| omega  | Omega dihedral in the backbone. A Harmonic constraint on planarity with standard deviation of ~6 deg. |
| p_aa_pp  | Probability of amino acid, given torsion values for phi and psi |
| fa_dun   | Internal energy of sidechain rotamers as derived from Dunbrack's statistics |
| yhh_planarity  | A special torsional potential to keep the tyrosine hydroxyl in the plane of the aromatic ring |
| ref   | Reference energy for each amino acid. Balances internal energy of amino acid terms.  Plays role in design. |
| METHOD_WEIGHTS   | Not an energy term itself, but the parameters for each amino acid used by the ref energy term.  |
