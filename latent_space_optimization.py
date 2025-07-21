from ase.io import read, write
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core import Lattice, Structure, Molecule
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core import PeriodicSite

from tqdm import tqdm
from pymatgen.core import Composition, Structure
from typing import List, Tuple

import re, joblib, json
import numpy as np
import pandas as pd
from keras.utils import to_categorical
from pyxtal.symmetry import Group
import pickle
import pandas as pd
from ase.io import read, write
from ase import Atoms
from ase.data import chemical_symbols
from ase.db import connect
import numpy as np
import sys, os
from spglib import get_spacegroup, find_primitive, standardize_cell
import pymatgen 
from pymatgen.io.ase import AseAtomsAdaptor
import ast
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error as mae
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

# Load the VAE model
model_dir = 'New_Kl5_ICSD_dis_site_middle_KL_element1000_lr_5e-06_epochs_2500_batch_64_test_0.2_val_0.1'
best_model = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if best_model:
    vae_dict = torch.load(model_dir+'/best_vae_model.pth',map_location=torch.device(device))
else: # use exit model
    vae_dict = torch.load(model_dir+'/exit_vae_model.pth',map_location=torch.device(device))

feature_dim = 183
wyckoff_dim = 9
crystal_dim = 236
kernel = [5,3,2]
stride = [2,3,2]
max_filter = 16
latent_dim = 256
vae_eval = VAE(feature_dim, wyckoff_dim, crystal_dim,verbose=False,kernel=kernel,stride=stride)
vae_eval.load_state_dict(vae_dict['model'])
vae_eval.to(device)

if 'Y_scaler' in vae_dict:
    scaler_Y = vae_dict['Y_scaler']
else:
    # Load the scaler if not present in the model dictionary
    print('Loading Y_scaler from file Y_scaler.gz')
    scaler_Y = joblib.load('Y_scaler.gz')

load_str = 'z_sample_NewICSD_train.npy'

# load the latent space
z_sample_train = np.load(load_str)

# Fit KDE 
kde = KernelDensity(kernel='gaussian', bandwidth=0.25)
kde.fit(z_sample_train)

# Assume the shape of z is [batch_size, latent_dim], where batch_size is the number of samples to optimize
# For example, let's optimize 1000 samples with 256-dimensional latent vectors
# Training parameters
batch_size = 10000
epochs = 4000
latent_dim = 256
coeff_element = 10
epsilon = 1e-8
lr = 0.01

# Define the conditions
element_list = ['Zn','V','O'] # elements
occupaction = [1,1,1] # occupaction

# Initialize z as a learnable parameter
z = torch.nn.Parameter(torch.tensor(kde.sample(n_samples=batch_size), dtype=torch.float32,device=device), requires_grad=True)

# Define the optimizer for z
optimizer = torch.optim.Adam([z], lr=lr)

### Element condition ###
target_element = torch.zeros(len(element_list),101,device=device) # 101 is the number of elements including vacancy
for i,ele in enumerate(element_list):
    target_element[i][chemical_symbols.index(ele)-1] = occupaction[i] # zero index
target_element[-1,-1] = 1 # include vacancy
target_element = target_element.repeat(batch_size,1, 1)
print('target_element:',target_element.shape)

target_element_all = torch.zeros(101,device=device) # 101 is the number of elements including vacancy
for i,ele in enumerate(element_list):
    target_element_all[chemical_symbols.index(ele)-1] = 1 # zero index
target_element_all[-1] = 1 # include vacancy
target_element_all = target_element_all.repeat(9, 1)
print('target_element_all:',target_element_all.shape)

# Optimization loop to modify latent samples z
for step in range(epochs):  # You can adjust the number of steps as needed
    optimizer.zero_grad()    
    # Use z directly to predict the space group (without decoding it)
    element,wyckoff_multiplier,frac_coords,wyckoff_letter, sg, lattice,disordered_site = vae_eval.decode(z)

    # Element loss
    element_loss = torch.zeros(batch_size,device=device) # Initialize element loss for each sample in the batch
    for i in range(batch_size):

        
        element_loss[i] = torch.square(element[i,:,:]-target_element_all).sum(axis=1).mean(axis=0)

        element_loss[i] += torch.square(element[i,:target_element.shape[1],:]-target_element[i]).sum(axis=1).mean(axis=0)
        
    if element_loss.shape[0] != 1:
        element_loss = torch.mean(element_loss)

    # Total loss
    loss = coeff_element*element_loss

    # Backpropagate to compute gradients
    loss.backward()
        
    # Update the latent vector z using gradient descent
    optimizer.step()
    # Print the loss and progress every 10 steps (or adjust as needed)
    if step % 10 == 0:
        print(f"Step {step}, Loss: {loss.item()}")

# predict SPG
z = z.cpu().detach().numpy()

# Save the optimized latent space
np.save('Decoded_data/z_samples_kde_optimized.npy', z)
