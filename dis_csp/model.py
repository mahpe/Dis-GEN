import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
import math, os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import json

class CrystalDataset(Dataset):
    def __init__(self, X, Y):
        """
        Initialize the dataset.
        
        Args:
        X (numpy array): Atomic features with shape (N, feature_dim, wyckoff_dim).
        Y (numpy array): Crystal features with shape (N, crystal_dim).
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        """
        Return the size of the dataset.
        """
        return len(self.X)

    def __getitem__(self, idx):
        """
        Retrieve an item at the specified index.
        
        Args:
        idx (int): Index of the item.
        
        Returns:
        Tuple[Tensor, Tensor]: Atomic and crystal features.
        """
        return self.X[idx], self.Y[idx]

######## VAE model #######################################################
# Define the VAE model
class VAE(nn.Module):
    def __init__(self, feature_dim, wyckoff_dim, crystal_dim,space_group_dim=230,lattice_dim=6,
                  stride= [2,2,1],kernel=[3,3,4],latent_dim=256, max_filter=64,verbose=False):
        super(VAE, self).__init__()
        self.feature_dim = feature_dim
        self.wyckoff_dim = wyckoff_dim
        self.crystal_dim = crystal_dim
        self.space_group_dim = space_group_dim
        self.stride = stride
        self.kernel_size = kernel
        self.latent_dim = latent_dim
        self.max_filter = max_filter
        self.verbose = verbose
        self.map_size = int(feature_dim / stride[0] / stride[1] / stride[2])

        # Encoder: Atomic features
        self.encoder_conv1 = nn.Conv1d(wyckoff_dim, max_filter * 2 , kernel_size=self.kernel_size[0], stride=self.stride[0], padding=0)
        self.encoder_conv2 = nn.Conv1d(max_filter * 2, max_filter * 4, kernel_size=self.kernel_size[1], stride=self.stride[1], padding=0)
        self.encoder_conv3 = nn.Conv1d(max_filter * 4, max_filter * 8, kernel_size=self.kernel_size[2], stride=self.stride[2], padding=0)

        self.encoder_bn1 = nn.BatchNorm1d(max_filter * 2)
        self.encoder_bn2 = nn.BatchNorm1d(max_filter * 4)
        self.encoder_bn3 = nn.BatchNorm1d(max_filter * 8)
        self.encoder_fc = nn.Linear(max_filter * 8 * (self.map_size) , 1024) #### TODO: NEED TO CHANGE THIS TO BE DYNAMIC

        # Encoder: Crystal features
        self.crystal_fc1 = nn.Linear(crystal_dim, 256)
        self.crystal_fc2 = nn.Linear(256, 128)

        # Latent space
        self.z_mean = nn.Linear(1024 + 128, latent_dim)
        self.z_log_var = nn.Linear(1024 + 128, latent_dim)

        # Decoder: Crystal features
        self.decoder_crystal_fc = nn.Linear(latent_dim, crystal_dim)
        self.decoder_sg = nn.Linear(crystal_dim, space_group_dim)  # 230 space groups
        self.decoder_lattice = nn.Linear(crystal_dim, lattice_dim) # 6 lattice constants and angles

        # Decoder: Atomistic features
        self.last_encoder_layer = self.encoder_conv3.state_dict()['weight'].shape[0]

        self.decoder_fc = nn.Linear(latent_dim, self.last_encoder_layer * self.map_size)
        self.decoder_conv1 = nn.ConvTranspose1d(self.last_encoder_layer, max_filter * 4, kernel_size=self.kernel_size[2], stride=self.stride[2])
        self.decoder_conv2 = nn.ConvTranspose1d(max_filter *4 , max_filter * 2 , kernel_size=self.kernel_size[1], stride=self.stride[1],)
        self.decoder_conv3 = nn.ConvTranspose1d(max_filter * 2, wyckoff_dim, kernel_size=self.kernel_size[0], stride=self.stride[0],)
        self.decoder_bn1 = nn.BatchNorm1d(max_filter * 4)
        self.decoder_bn2 = nn.BatchNorm1d(max_filter * 2)
        self.decoder_bn3 = nn.BatchNorm1d(wyckoff_dim)

        self.element = nn.Linear(feature_dim,101)
        self.wycokff_letter = nn.Linear(feature_dim,27)
        self.wyckoff_multiplier = nn.Linear(feature_dim,51)
        self.frac_coords = nn.Linear(feature_dim,3)
        self.disordered_site = nn.Linear(feature_dim,1)

    def encode(self, x, x2):
        # Atomic feature encoding
        if self.verbose:
            print('X:',x.shape)
            print('X2:',x2.shape)
        #x = x.permute(0, 2, 1)
        #print(x.shape,x2.shape)
        en0 = F.relu(self.encoder_bn1(self.encoder_conv1(x)))
        if self.verbose: print('En0:',en0.shape)     
        en1 = F.relu(self.encoder_bn2(self.encoder_conv2(en0)))
        if self.verbose: print('En1:',en1.shape)
        en2 = F.relu(self.encoder_bn3(self.encoder_conv3(en1)))
        if self.verbose: print('En2:',en2.shape)
        en3 = en2.view(en2.size(0), -1)  # Flatten
        if self.verbose: print('En3:',en3.shape)

        # Get the map size
        self.map_size = en2.shape[-1]

        en4 = F.relu(self.encoder_fc(en3))
        if self.verbose: print('En4:',en4.shape)
        if self.verbose: print('---------------------------------')

        # Crystal feature encoding

        cry0 = F.relu(self.crystal_fc1(x2))
        if self.verbose: print('Cry0:',cry0.shape)
        cry1 = F.relu(self.crystal_fc2(cry0))
        if self.verbose: print('Cry1:',cry1.shape)
        if self.verbose: print('---------------------------------')

        # Combine atomic and crystal features
        latent = torch.cat((en4, cry1), dim=1)
        z_mean = self.z_mean(latent)
        z_log_var = self.z_log_var(latent)

        if self.verbose:
            print('Latent indput',latent.shape)
            print('Z mean',z_mean.shape,'Z_std',z_log_var.shape)
            print('---------------------------------')

        return z_mean, z_log_var

    def sampling(self, z_mean, z_log_var):
        std = torch.exp(0.5 * z_log_var)
        eps = torch.randn_like(std)
        return z_mean + eps * std

    def decode(self, z):
        # Decode crystal features
        # Note for features with cross entropy loss, we do not apply softmax activation (ref: https://stackoverflow.com/questions/55675345/should-i-use-softmax-as-output-when-using-cross-entropy-loss-in-pytorch)
        if self.verbose:
            print('Z:',z.shape)

        crystal_features = F.relu(self.decoder_crystal_fc(z))
        decoded_sg = self.decoder_sg(crystal_features)
        decoded_lattice = self.decoder_lattice(crystal_features)


        if self.verbose:
            print('Crystal Features:',crystal_features.shape)
            print('Decoded SG:',decoded_sg.shape)
            print('Decoded Lattice:',decoded_lattice.shape)
            print('---------------------------------')

        # Decode atomistic features
        dec0 = self.decoder_fc(z)
        if self.verbose: print('Dec0:',dec0.shape)
        dec1 = dec0.view(dec0.size(0), self.last_encoder_layer ,self.map_size)
        if self.verbose: print('Dec1:',dec1.shape)
        dec2 = F.relu(self.decoder_bn1(self.decoder_conv1(dec1)))
        if self.verbose: print('Dec2:',dec2.shape)
        dec3 = F.relu(self.decoder_bn2(self.decoder_conv2(dec2)))
        if self.verbose: print('Dec3:',dec3.shape)

        dec4= F.relu(self.decoder_bn3(self.decoder_conv3(dec3)))
        if self.verbose: print('Dec4:',dec4.shape)

        # Separate the atomistic features. 
        decoded_element = F.softmax(self.element(dec4),dim=2)
        if self.verbose: print('Element:',decoded_element.shape)
        decoded_wyckoff_multiplier = self.wyckoff_multiplier(dec4)
        if self.verbose: print('Wyckoff Multiplier:',decoded_wyckoff_multiplier.shape)
        decoded_frac_coords = self.frac_coords(dec4)
        if self.verbose: print('Frac Coords:',decoded_frac_coords.shape)
        decoded_wyckoff_letter =self.wycokff_letter(dec4)
        if self.verbose: print('Wyckoff Letter:',decoded_wyckoff_letter.shape)
        decoded_disordered_site = self.disordered_site(dec4)
        if self.verbose: print('Disordered Site:',decoded_disordered_site.shape)
        
        
        return decoded_element,decoded_wyckoff_multiplier,decoded_frac_coords,decoded_wyckoff_letter, decoded_sg, decoded_lattice,decoded_disordered_site

    def forward(self, x, x2):
        # Encode
        z_mean, z_log_var = self.encode(x, x2)
        # Latent space sampling
        z = self.sampling(z_mean, z_log_var)
        # Decode
        decoded_element,decoded_wyckoff_multiplier,decoded_frac_coords,decoded_wyckoff_letter, decoded_sg, decoded_lattice,decoded_disordered_site = self.decode(z)
        outputs = {'decoded_element':decoded_element,
                    'decoded_wyckoff_multiplier':decoded_wyckoff_multiplier,
                    'decoded_frac_coords':decoded_frac_coords,
                    'decoded_wyckoff_letter':decoded_wyckoff_letter,
                    'decoded_disordered_site':decoded_disordered_site,
                   'decoded_sg':decoded_sg,
                   'decoded_lattice':decoded_lattice,
                   'z_mean':z_mean,
                   'z_log_var':z_log_var}
        return outputs