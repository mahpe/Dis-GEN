from pymatgen.core import  Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from tqdm import tqdm
from pymatgen.core import Composition, Structure
from typing import List, Tuple

import re, joblib, json
import numpy as np
from pyxtal.symmetry import Group
import pickle
import pandas as pd
import numpy as np
import sys, os
import ast
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import roc_curve, auc,accuracy_score

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score # for cross-validation
from datetime import datetime 
from sklearn.preprocessing import LabelEncoder,OneHotEncoder, StandardScaler
from pymatgen.core import Composition, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.cif import CifParser
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pymatgen.io.cif import CifWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
import math
from tqdm import tqdm
from sklearn.neighbors import KernelDensity

sys.path.append('/home/energy/mahpe/Published_code/Dis-CSP/dis_csp')

from dis_csp.model import CrystalDataset, VAE
from dis_csp.generation import decode_samples, generate_wyckoffgene, get_cif_lines
from dis_csp.csp_filter import structure_validity,oxidation_state_validity
SG_SYM = {spacegroup: Group(spacegroup) for spacegroup in range(1, 231)}
SG_TO_WP_TO_SITE_SYMM = dict()
for spacegroup in range(1, 231):
    SG_TO_WP_TO_SITE_SYMM[spacegroup] = dict()
    for wp in SG_SYM[spacegroup].Wyckoff_positions:
        wp_site = str(wp.multiplicity)+wp.letter
        wp.get_site_symmetry()
        SG_TO_WP_TO_SITE_SYMM[spacegroup][wp_site] = wp
#       SG_TO_WP_TO_SITE_SYMM[spacegroup][wp_site][wp] = wp.get_site_symmetry_object().to_one_hot()

latent_space_path = '/home/energy/mahpe/Published_code/Dis-CSP/Decoded_data/z_samples_kde.npy'
model_dir = 'New_Kl5_ICSD_dis_site_middle_KL_element1000_lr_5e-06_epochs_2500_batch_64_test_0.2_val_0.1'
latent_space_path = 'Decoded_data/z_sample_NewICSD_test.npy'
cif_save_path = 'Decoded_data/Generated_cif'
best_model = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### Load the VAE model ###
if best_model:
    vae_dict = torch.load(model_dir+'/best_vae_model.pth',map_location=torch.device('cpu'))
else: # use exit model
    vae_dict = torch.load(model_dir+'/exit_vae_model.pth',map_location=torch.device('cpu'))

feature_dim = 183
wyckoff_dim = 9
crystal_dim = 236
kernel = [5,3,2]
stride = [2,3,2]
max_filter = 16
latent_dim = 256
vae_eval = VAE(feature_dim, wyckoff_dim, crystal_dim,verbose=False,kernel=kernel,stride=stride)
vae_eval.load_state_dict(vae_dict['model'])

# Decode the latent space samples
print(f"Loading latent space samples from {latent_space_path}")
z_samples = np.load(latent_space_path)

if 'Y_scaler' in vae_dict:
    scaler_Y = vae_dict['Y_scaler']
else:
    # Load the scaler if not present in the model dictionary
    print('Loading Y_scaler from file Y_scaler.gz')
    scaler_Y = joblib.load('Y_scaler.gz')

### Load the latent space samples ###
z_samples = np.load(latent_space_path)

# Get the decoded samples
save_dict = decode_samples(z_samples,vae_eval,scaler_Y,device='cpu')


# Generate Wyckoff genes
mixiter = None
element_acc = 0.05  
disorder_acc = 0.5
all_wyckoffgenes,sma_int, failed_int = generate_wyckoffgene(save_dict,
                             max_iter=mixiter,
                             element_acc=element_acc,
                             disorder_acc=disorder_acc,
                             shift_frac_coord=True,
                             verbose=False)
if mixiter is not None:
    print(f"Symmetry Matching Accuracy: {100-sma_int/mixiter*100:.2f}%")
    print(f"Number of failed items: {failed_int} of {mixiter} processed, Procentage: {failed_int/mixiter*100:.2f}%")
else:
    print(f"Symmetry Matching Accuracy: {100- sma_int/len(save_dict['spacegroup'])*100:.2f}%")
    print(f"Number of failed items: {failed_int} of {len(save_dict['spacegroup'])} processed, Procentage: {failed_int/len(save_dict['spacegroup'])*100:.2f}%")

all_cif_lines = []
cif_lines_wyckoffgene = []
failed_count = 0
validity_count = 0
oxidation_state_count = 0
validity_count_list = []

maxiter = None

for i in tqdm(range(len(all_wyckoffgenes))):
    if maxiter is not None:
        if i >= maxiter:
            break
    wyckoffgene = all_wyckoffgenes[i]
    cif_lines = get_cif_lines(wyckoffgene)
    try:
        structure = Structure.from_str(cif_lines,fmt='cif')
        sga = SpacegroupAnalyzer(structure,symprec=0.1)
        refined_struc = sga.get_refined_structure()
        sga = SpacegroupAnalyzer(refined_struc,symprec=0.01)
        structure = sga.get_symmetrized_structure()
    except:
        failed_count +=1
        continue
    
    struc_valid =structure_validity(structure)
    oxidation_valid = oxidation_state_validity(structure,two_oxidation_state=False,verbose=False)

    if not struc_valid:
        validity_count += 1
    if not oxidation_valid:
        oxidation_state_count += 1
    # if one of them is not valid, skip the structure
    if not (struc_valid and oxidation_valid):
        continue
    # Add the structure to the list
    all_cif_lines.append(cif_lines)
    cif_lines_wyckoffgene.append(wyckoffgene)
    
if maxiter is not None:
    print(f'Failed to parse {failed_count} structures out of {maxiter} Procentage: {failed_count/maxiter*100:.2f}%')
    print(f'Validity check failed for {validity_count} structures out of {maxiter} Procentage: {validity_count/maxiter*100:.2f}%')
    print(f'Oxidation state check failed for {oxidation_state_count} structures out of {maxiter} Procentage: {oxidation_state_count/maxiter*100:.2f}%')
    print(f'Total valid structures: {len(all_cif_lines)} out of {maxiter} Procentage: {len(all_cif_lines)/maxiter*100:.2f}%')
else:
    print(f"Failed to parse {failed_count} structures out of {len(all_wyckoffgenes)} Procentage: {failed_count/len(all_wyckoffgenes)*100:.2f}%")
    print(f"Validity check failed for {validity_count} structures out of {len(all_wyckoffgenes)} Procentage: {validity_count/len(all_wyckoffgenes)*100:.2f}%")
    print(f"Oxidation state check failed for {oxidation_state_count} structures out of {len(all_wyckoffgenes)} Procentage: {oxidation_state_count/len(all_wyckoffgenes)*100:.2f}%")
    print(f"Total valid structures: {len(all_cif_lines)} out of {len(all_wyckoffgenes)} Procentage: {len(all_cif_lines)/len(all_wyckoffgenes)*100:.2f}%")

# make a dictionary with the cif lines and the indexes
cif_dict = {}
for i, cif_lines in enumerate(all_cif_lines):
    cif_dict[i] = cif_lines

cif_dict_name = latent_space_path.split('/')[-1].replace('.npy','_cif_dict.json')

# Save the cif_dict to a file
cif_dict_path = os.path.join(cif_save_path, cif_dict_name)
with open(cif_dict_path, 'w') as f:
    json.dump(cif_dict, f)

# Save the cif lines to a file
cif_file_path = os.path.join(cif_save_path, latent_space_path.split('/')[-1].replace('.npy',''))
if not os.path.exists(cif_file_path):
    os.makedirs(cif_file_path)


for i, cif_lines in enumerate(all_cif_lines):
    composition = cif_lines_wyckoffgene[i]['structure_pymatgen'].composition
    cif_file_name = f'{i}_{composition}_spg_{cif_lines_wyckoffgene[i]["spacegroup"]}.cif'
    with open(os.path.join(cif_file_path,cif_file_name), 'w') as f:
        f.write(cif_lines)