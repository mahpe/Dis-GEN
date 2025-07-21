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
from matplotlib.colors import LogNorm
from sklearn.metrics import roc_curve, auc,accuracy_score
from sklearn.preprocessing import label_binarize
from sklearn.mixture import GaussianMixture

from pyxtal.symmetry import Group
from sklearn.neighbors import KernelDensity
import torch

sys.path.append('/home/energy/mahpe/Published_code/Dis-CSP/dis_csp')
from dis_csp.model import VAE
# Load the VAE model
model_dir = 'New_Kl5_ICSD_dis_site_middle_KL_element1000_lr_5e-06_epochs_2500_batch_64_test_0.2_val_0.1'
best_model = True

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

## Load the train latents space
z_sample_train = np.load('z_sample_NewICSD_train.npy')

# Estimate latent space distribution
batch_size = 27008 #10000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if 'Y_scaler' in vae_dict:
    scaler_Y = vae_dict['Y_scaler']
else:
    # Load the scaler if not present in the model dictionary
    print('Loading Y_scaler from file Y_scaler.gz')
    scaler_Y = joblib.load('Y_scaler.gz')

# KDE for latent space
kde = KernelDensity(kernel='gaussian', bandwidth=0.25)
kde.fit(z_sample_train)
# Sample from the learned distribution
z_samples_kde = torch.tensor(kde.sample(n_samples=batch_size)).float().to(device)
np.save('Decoded_data/z_samples_kde.npy', z_samples_kde.cpu().numpy())

# GMM for latent space
gmm = GaussianMixture(n_components=5, covariance_type='full')
gmm.fit(z_sample_train)

# Sample from the learned distribution
z_samples_gmm = torch.tensor(gmm.sample(n_samples=batch_size)[0]).float().to(device)
np.save('Decoded_data/z_samples_gmm.npy', z_samples_gmm.cpu().numpy())

# Random sampling from latent space
z_sample_ran = torch.randn(batch_size, latent_dim)
np.save('Decoded_data/z_samples_ran.npy', z_sample_ran.cpu().numpy())