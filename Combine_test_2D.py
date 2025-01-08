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
from sklearn.model_selection import train_test_split
import pandas as pd
import json

##### Load data ########################################################
test_size = 0.2
val_size = 0.1
batch_size = 32
log_step = 1
lr = 0.0001
epochs = 200
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

folder_name = 'normal'
folder_name = folder_name + f'lr_{lr}_epochs_{epochs}_batch_{batch_size}_test_{test_size}_val_{val_size}'

if not os.path.exists(folder_name):
    os.makedirs(folder_name)

#X = crystal_rep.total_atomistic_features
#Y = crystal_rep.total_crystal_features
#np.save('X_2D.npy',X)
#np.save('Y_2D.npy',Y)
X = np.load('X_2D.npy')
Y = np.load('Y_2D.npy')

print('X:',X.shape)
print('Y:',Y.shape)

# Setup scaler for Y
scaler_Y = StandardScaler()
Y_scaled = Y.copy()
Y_scaled[:,:6] = scaler_Y.fit_transform(Y[:,:6])
# Set X and Y to torch tensors
Y = Y_scaled

x_train_val,x_test,y_train_val,y_test=train_test_split( X \
                                               ,Y,test_size=test_size)
x_train,x_val,y_train,y_val = train_test_split(x_train_val,y_train_val,test_size=val_size)
print('Train:',x_train.shape,y_train.shape)
print('Val:',x_val.shape,y_val.shape)
print('Test:',x_test.shape,y_test.shape)

# 2. Define a custom Dataset class
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

# 3. Create Dataset and DataLoader for each split
train_dataset = CrystalDataset(x_train, y_train)
val_dataset = CrystalDataset(x_val, y_val)
#test_dataset = CrystalDataset(x_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
#test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

######## VAE model #######################################################
# Define the VAE model
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
        
        
        return decoded_element,decoded_wyckoff_multiplier,decoded_frac_coords,decoded_wyckoff_letter, decoded_sg, decoded_lattice

    def forward(self, x, x2):
        # Encode
        z_mean, z_log_var = self.encode(x, x2)
        # Latent space sampling
        z = self.sampling(z_mean, z_log_var)
        # Decode
        decoded_element,decoded_wyckoff_multiplier,decoded_frac_coords,decoded_wyckoff_letter, decoded_sg, decoded_lattice = self.decode(z)
        outputs = {'decoded_element':decoded_element,
                    'decoded_wyckoff_multiplier':decoded_wyckoff_multiplier,
                    'decoded_frac_coords':decoded_frac_coords,
                    'decoded_wyckoff_letter':decoded_wyckoff_letter,
                   'decoded_sg':decoded_sg,
                   'decoded_lattice':decoded_lattice,
                   'z_mean':z_mean,
                   'z_log_var':z_log_var}
        return outputs

# Initialize the VAE
feature_dim = X.shape[2]
wyckoff_dim = X.shape[1]
crystal_dim = Y.shape[1]
kernel = [4,3,2]
stride = [2,3,2]
max_filter = 32
latent_dim = 256

vae = VAE(feature_dim, wyckoff_dim, crystal_dim,verbose=True,kernel=kernel,
          stride=stride,max_filter=max_filter,latent_dim=latent_dim)
# Test the forward pass
x_batch, y_batch = next(iter(train_loader))
out = vae(x_batch, y_batch)

# NOTE: Kernel add an extra dimension for the wyckoff sites 
#from torchsummary import summary1
#summary(vae, [(feature_dim,wyckoff_dim),(crystal_dim,)],)
# 
############################################################################

##### Loss function ######################################################


# Defien loss function and test it
# Define the VAE loss function

# Defien loss function and test it
# Define the VAE loss function
def vae_loss_function(model_output, x, y, coeffs,verbose=False):
    """
    Compute the VAE loss.

    Args:
    reconstructed_x (Tensor): Reconstructed output from the decoder.
    x (Tensor): Original input.
    z_mean (Tensor): Mean of the latent variable.
    z_log_var (Tensor): Log variance of the latent variable.
    coeffs (dict): Coefficients for different loss components.

    Returns:
    Tensor: Total loss (scalar).
    """   
    # VAE loss consists of two components: 
    # 1. KL divergence loss. That is a measure of divergence between two distributions.
    # 2. Reconstruction loss for the atoms. That we will seperate into several lossed for each component we want to reconstruct
    # 3. Reconstruction loss for the crystal features
    # We want to minimize the reconstruction loss and the KL divergence loss.
    # Mean over the batch to make the training independent of the batch size.

    # Initialize the losses
    losses = {}

    # Get the model output
    z_mean = model_output['z_mean']
    z_log_var = model_output['z_log_var']
    decoded_sg = model_output['decoded_sg']
    decoded_lattice = model_output['decoded_lattice']

    # Get the decoded atomistic features
    decoded_element = model_output['decoded_element']
    decoded_wyckoff_multiplier = model_output['decoded_wyckoff_multiplier']
    decoded_frac_coords = model_output['decoded_frac_coords']
    decoded_wyckoff_letter = model_output['decoded_wyckoff_letter']

    if verbose:
        print('Original:',x.shape)
        print('Reconstructed:',decoded_element.shape,decoded_wyckoff_multiplier.shape,decoded_frac_coords.shape,decoded_wyckoff_letter.shape)
        print('Z mean:',z_mean.shape)
        print('Z log var:',z_log_var.shape)
        print('Coeffs:',coeffs)
        print('---------------------------------')
    
    #### KL Divergence loss ####
    kl_loss_i = (1 + z_log_var - torch.square(z_mean) - torch.exp(z_log_var))
    kl_loss = torch.mean( -0.5 * (1/z_mean.shape[1]) * torch.sum(kl_loss_i, dim=1) ) # sum over the latent dimensions and mean over the batch
    #losses['kl'] = kl_loss
    if verbose:
        print('KL Loss i:',kl_loss_i.shape)
        print('KL Loss:',kl_loss.shape)

    #### Reconstruction loss atomistic features ####
    #if verbose:
    #    if decoded_element[0,0,:].sum() < 1 or decoded_element[0,0,:].sum() > 1:
    #        print('ERROR Element:',decoded_element[0,0,:].sum())
    
    # Loop over wyckoff sites
    element_loss_i = []
    wyckoff_site_loss_i = []
    wyckoff_multiplier_loss_i = []
    for i in range(decoded_element.shape[1]):
        # Reconstruction loss of the elemental features
        # Use KL div loss to predict the prob. distribution (ref: https://discuss.pytorch.org/t/loss-function-for-predicting-a-distribution/156681)
        element_loss_i.append(torch.mean(torch.square(decoded_element[:,i,:] - x[:,i,:101]),dim=1)) # MSE between the softmax distribution and the true distribution
        #element_loss_i.append(torch.sum(F.kl_div(F.log_softmax(decoded_element[:,i,:],dim=1), x[:,i,:101],reduction='none'),dim=1)) # KL divergence loss with sum reduction

        # Wyckoff sites reconstruction loss. Cross entropy loss
        wyckoff_site_loss_i.append(F.cross_entropy(decoded_wyckoff_letter[:,i,:], x[:,i,-27:],reduction='none'))
        #wyckoff_site_loss_i.append(torch.mean(torch.square(decoded_wyckoff_letter[:,i,:], x[:,i,-27:]),dim=1)) # MSE between the softmax distribution and the true distribution)
        #wyckoff_site_loss_i.append(torch.square(torch.argmax(decoded_wyckoff_letter[:,i,:],dim=1) - torch.argmax(x[:,i,-27:],dim=1) ) ) # MSE between the softmax distribution and the true distribution)


        # Wyckoof multiplier loss. Cross entropy loss
        wyckoff_multiplier_loss_i.append(F.cross_entropy(decoded_wyckoff_multiplier[:,i,:], x[:,i,101:-30],reduction='none',))
    # Element loss
    element_loss_i = torch.stack(element_loss_i)
    element_loss = torch.mean(torch.sum(element_loss_i,dim=0)) # sum over the wyckoff_sites and mean over the batch
    losses['element'] = element_loss
    if verbose:
        print('Element Loss i:',element_loss_i.shape)
        print('Element Loss:',element_loss.shape)

    # Wyckoff sites reconstruction loss. Cross entropy loss
    wyckoff_site_loss_i = torch.stack(wyckoff_site_loss_i)
    wyckoff_site_loss = torch.mean(torch.sum(wyckoff_site_loss_i,dim=0,dtype=torch.float64)) # sum over the wyckoff sites and mean over the batch
    losses['wyckoff_letter'] = wyckoff_site_loss
    if verbose:
        print('Wyckoff Loss i:',wyckoff_site_loss_i.shape)
        print('Wyckoff Loss:',wyckoff_site_loss.shape)

    # Wyckoof multiplier loss. Cross entropy loss
    wyckoff_multiplier_loss_i = torch.stack(wyckoff_multiplier_loss_i)
    wyckoff_multiplier_loss = torch.mean(torch.sum(wyckoff_multiplier_loss_i,dim=0)) # sum over the wyckoff sites and mean over the batch
    losses['wyckoff_multiplier'] = wyckoff_multiplier_loss
    if verbose:
        print('Wyckoff Multiplier Loss i:',wyckoff_multiplier_loss_i.shape)
        print('Wyckoff Multiplier Loss:',wyckoff_multiplier_loss.shape)
    
    # Fractional coordinates reconstruction loss. Mean squared error loss
    frac_coords_loss_i = torch.mean(torch.square(decoded_frac_coords - x[:,:,-30:-27]),dim=2) # mse over the wyckoff sites 
    frac_coords_loss_j = torch.sum(frac_coords_loss_i,dim=1) # sum over the fractional coordinates
    frac_coords_loss = torch.mean(frac_coords_loss_j) # mean over the batch
    losses['frac_coords'] = frac_coords_loss
    if verbose:
        print('Frac Coords Loss i:',frac_coords_loss_i.shape)
        print('Frac Coords Loss:',frac_coords_loss.shape)

    #### Reconstruction loss crystal features ####
    # Space group loss. Cross entropy loss
    space_group_loss_i = F.cross_entropy(decoded_sg, y[:,6:],reduce=False)
    space_group_loss = torch.mean(space_group_loss_i) # mean over the batch
    losses['space_group'] = space_group_loss
    if verbose:
        print('Space Group Loss i:',space_group_loss_i.shape)
        print('Space Group Loss:',space_group_loss.shape)

    # Lattice loss. Mean squared error loss
    lattice_loss_i = F.mse_loss(decoded_lattice, y[:,:6],reduce=False)
    lattice_loss = torch.mean(torch.sum(lattice_loss_i,dim=1)) # sum over the lattice constants and mean over the batch
    losses['lattice'] = lattice_loss
    if verbose:
        print('Lattice Loss i:',lattice_loss_i.shape)
        print('Lattice Loss:',lattice_loss.shape)

    # Weighted sum of losses
    for key in losses:
        losses[key] *= coeffs[key]

    # Total loss
    total_loss = torch.sum(torch.stack(list(losses.values()))) # Take all the losses and make them to a list of torch objects then stack them to make them a tensor and sum them
    losses['total'] = total_loss
    if verbose:
        print('Total Loss:',total_loss.shape)
        print('---------------------------------')

    return losses
coeffs = {'kl': 1000.0,
          'element':200.0,
          'wyckoff_letter': 1.0,
        'wyckoff_multiplier': 1.0,
          'frac_coords': 1.0,
          'space_group':10.0,
          'lattice':3.0}

# Forward pass
vae = VAE(feature_dim, wyckoff_dim, crystal_dim,verbose=False,kernel=kernel,stride=stride)
out = vae(x_batch, y_batch)

# Compute loss
loss = vae_loss_function(out, x_batch, y_batch, coeffs,verbose=True);
for key in loss:
    print(f"{key}: {loss[key]}")

########################################################

########### Training loop ################################
# Training loop
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#log_step = 1
#epochs = 50
#lr = 0.001
# Initialize the VAE
feature_dim = X.shape[2]
wyckoff_dim = X.shape[1]

crystal_dim = Y.shape[1]
vae = VAE(feature_dim, wyckoff_dim, crystal_dim,verbose=False,kernel=kernel,stride=stride)
vae.to(device)

optimizer = Adam(vae.parameters(), lr=lr)

# Initialize the train and validation loss
train_loss_dict = {}
train_loss_dict['total'] = []
train_loss_dict['epoch'] = []

val_loss_dict = {}
val_loss_dict['total'] = []
val_loss_dict['epoch'] = []

best_val_loss = np.inf

# Training loop
for epoch in tqdm(range(epochs)):
    vae.train()
    # Initialize the train loss
    train_loss = 0.0
    train_running_loss = {}

    # Loop through the train loader
    for x_batch, y_batch in train_loader:  # Use both datasets if required
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()

        # Forward pass
        out = vae(x_batch, y_batch)
        
        # Compute loss
        loss = vae_loss_function(out, x_batch, y_batch, coeffs)
        # Backward pass
        loss['total'].backward()
        optimizer.step()

        # Log batch results
        train_loss += loss['total'].item()
        # Log running loss for each component of the loss
        if len(train_running_loss) == 0:
            train_running_loss = loss
        else:
            for key in loss:
                train_running_loss[key] += loss[key]

    print(f"Epoch {epoch + 1}/{epochs}: Train Loss: {train_loss / len(train_loader)}")
    
    # Save the train loss for each component of the loss as a mean over the batches
    for key in train_running_loss:
        if key not in train_loss_dict:
            train_loss_dict[key] = [train_running_loss[key].item() / len(train_loader)]
        else:
            train_loss_dict[key].append(train_running_loss[key].item() / len(train_loader))
    train_loss_dict['epoch'].append(epoch)
    
    # Validate and save model for each log step
    if (epoch + 1) % log_step == 0:
        vae.eval()
        val_loss = 0.0
        val_running_loss = {}
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                # Forward pass
                out_val = vae(x_val, y_val)
                
                # Compute loss
                loss_val = vae_loss_function(out_val, x_val, y_val,coeffs)

                # Log batch results for validation 
                if len(val_running_loss) == 0:
                    val_running_loss = loss_val
                else:
                    for key in loss_val:
                        val_running_loss[key] += loss_val[key]
        # Log epoch results
        print('---------------------------------')
        print(f"Validation Loss:")
        
        for key in val_running_loss:
            # Print all losses
            print(f"{key}: {val_running_loss[key].item()/len(val_loader)}")
            # Save the validation loss
            if key not in val_loss_dict:
                val_loss_dict[key] = [val_running_loss[key].item()/len(val_loader)]
            else:
                val_loss_dict[key].append(val_running_loss[key].item()/len(val_loader))
        val_loss_dict['epoch'].append(epoch)


        # Save the model if the validation loss is the best
        if val_running_loss['total'] < best_val_loss:
            best_val_loss = val_running_loss['total']
            torch.save({"model": vae.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "coeffs": coeffs,
                        "epoch": epoch,
                        "total loss": best_val_loss,
                        "val loss": val_loss_dict,
                        "train loss": train_loss_dict,
                        },      
                        f'{folder_name}/best_vae_model.pth')
            print('Model saved')

###### Plot the losses ##############################################
torch.save({"model": vae.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "coeffs": coeffs,
                        "epoch": epoch,
                        "total loss": best_val_loss,
                        "val_loss": val_loss_dict,
                        "train_loss": train_loss_dict,
                        },      
                       f'{folder_name}/exit_vae_model.pth')

## Plot all the losses as a function of the epoch
fig, ax = plt.subplots(1,2,figsize=(20,10))
font_size = 20
Nbins = 20

for key in train_loss_dict:
    if key == 'epoch':
        continue
    ax[0].plot(train_loss_dict[key],label=key)
    val_epoch  = np.arange(0,len(val_loss_dict[key])*log_step,log_step)
    ax[1].plot(val_epoch, val_loss_dict[key],label=key)

ax[0].set_xlabel('Epoch',fontsize=font_size)
ax[0].set_ylabel('Loss',fontsize=font_size)
ax[0].set_title('Train Loss',fontsize=font_size)
ax[0].tick_params(axis='both', which='major', labelsize=font_size)
ax[0].tick_params(axis='both', which='minor', labelsize=font_size)
ax[0].grid(True)
ax[0].legend(fontsize=font_size,ncol=2)

ax[1].set_xlabel('Epoch',fontsize=font_size)
ax[1].set_ylabel('Loss',fontsize=font_size)
ax[1].set_title('Validation Loss',fontsize=font_size)
ax[1].tick_params(axis='both', which='major', labelsize=font_size)
ax[1].tick_params(axis='both', which='minor', labelsize=font_size)
ax[1].grid(True)
#ax[1].legend(fontsize=font_size)

# Save the figure
plt.savefig(f'{folder_name}/Losses.png')

# Save loss coefficients
with open(f'{folder_name}/coeffs.json', 'w') as f:
    json.dump(coeffs, f)

# Save the losses
train_loss_df = pd.DataFrame(train_loss_dict)
val_loss_df = pd.DataFrame(val_loss_dict)
train_loss_df.to_csv(f'{folder_name}/train_loss.csv')
val_loss_df.to_csv(f'{folder_name}/val_loss.csv')
