import numpy as np
import torch
from dis_gen.model import VAE
from dis_gen.generation import decode_samples, generate_wyckoffgene, generate_cif_files
import joblib, json, os
import warnings
warnings.filterwarnings("ignore")

latent_space_path = 'Decoded_data/z_samples_gmm.npy'
#latent_space_path = 'Decoded_data/z_samples_ran.npy'
#latent_space_path = 'Decoded_data/z_samples_kde.npy'
#latent_space_path = 'Decoded_data/z_sample_NewICSD_test.npy'
model_dir = 'New_Kl5_ICSD_dis_site_middle_KL_element1000_lr_5e-06_epochs_2500_batch_64_test_0.2_val_0.1'
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
#latent_space_path = 'Decoded_data/z_samples_kde_optimized.npy'
cif_save_path = 'Decoded_data/Generated_cif'
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
print('Decoded samples shape:', len(save_dict['spacegroup']))

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
    print(f"Number of failed items: {failed_int} of {mixiter} processed, Procentage: {100-failed_int/mixiter*100:.2f}%")
else:
    print(f"Symmetry Matching Accuracy: {100- sma_int/len(save_dict['spacegroup'])*100:.2f}%")
    print(f"Number of failed items: {failed_int} of {len(save_dict['spacegroup'])} processed, Procentage: {100-failed_int/len(save_dict['spacegroup'])*100:.2f}%")

all_cifs,cifs__wyckoffgene = generate_cif_files(all_wyckoffgenes,maxiter=None,validity_primitive=False,symmetry_analyzer=False, verbose=False, two_oxidation_state=False)

# make a dictionary with the cif lines and the indexes
cif_dict = {}
for i, cif_lines in enumerate(all_cifs):
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


for i, cif_lines in enumerate(all_cifs):
    composition = cifs__wyckoffgene[i]['structure_pymatgen'].composition
    cif_file_name = f'{i}_{composition}_spg_{cifs__wyckoffgene[i]["spacegroup"]}.cif'
    with open(os.path.join(cif_file_path,cif_file_name), 'w') as f:
        f.write(cif_lines)