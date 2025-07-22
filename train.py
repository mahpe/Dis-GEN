import os 
import numpy as np
import math
import json, sys, toml
import argparse
import logging
import itertools
import torch
import time
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

from dis_gen.model import CrystalDataset, VAE
from dis_gen.loss_function import vae_loss_function
from dis_gen.utils import split_data
# Define function to get arguments
def get_arguments(arg_list=None):
    parser = argparse.ArgumentParser(
        description="Train graph convolution network", fromfile_prefix_chars="+"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory to save the output files. Default is 'output'.",
    )
    parser.add_argument(
        "--dataset_atomic",
        type=str,
        help="Path to the atomic representation of the dataset(e.g., .npy or .npz).",
    )
    parser.add_argument(
        "--dataset_crystal",
        type=str,
        help="Path to the crystal representation of the dataset(e.g., .npy or .npz).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for training. Default is 64.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs for training. Default is 100.",
    )
    parser.add_argument(
        "--log_step",
        type=int,
        default=10,
        help="Number of epochs between logging the training and validation loss. Default is 10.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate for the optimizer. Default is 0.001.",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Proportion of the dataset to include in the test split. Default is 0.2.",
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=0.1,
        help="Proportion of the dataset to include in the validation split. Default is 0.1.",
    )
    parser.add_argument(
        "--split_file",
        type=str,
        default=None,
        help="Path to the JSON file containing precomputed dataset splits. If provided, will override test_size and val_size.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility. Default is 42.",
    )
    parser.add_argument(
        "--kernel",
        type=list,
        help="Kernel size for the convolutional layers. Must be fitted to the input data shape.",
    )
    parser.add_argument(
        "--stride",
        type=list,
        help="Stride for the convolutional layers. Must be fitted to the input data shape.",
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=256,
        help="Dimensionality of the latent space. Default is 256.",
    )
    parser.add_argument(
        "--max_filter",
        type=int,
        default=16,
        help="Maximum number of filters in the convolutional layers. Default is 64.",
    )
    parser.add_argument(
        "--loss_weights",
        type=dict,
        default=None,
        help="Weights for the loss function. Default is None.",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default='config.toml',
        help="Path to the configuration file (TOML format). If provided, will override command line arguments.",
    )
    parser.add_argument(
        "--verbose",
        type=bool,
        help="Enable verbose in training. Default is False.",
        default=False,
    )

    return parser.parse_args(arg_list)
def update_namespace(ns:argparse.Namespace, d:dict) -> None:
    """

    Update the namespace with the dictionary.

    Args:
        ns: The namespace to update
        d: The dictionary to update the namespace with
    
    """
    for k, v in d.items():
        
        ns.__dict__[k] = v
def main():
    # Load argument Namespace
    args = get_arguments()

    # Load configuration file if provided
    if args.config_file:
        with open(args.config_file, 'r') as f:
            config = toml.load(f)
        # Update args with config values
        update_namespace(args, config)

    # Ensure output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # Set logging configuration
    logging.basicConfig(
        filename=os.path.join(args.output_dir, 'training.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.info("Starting training with arguments: %s", args)

    # Check if kernel and stride are provided
    if args.kernel is None or args.stride is None:
        raise ValueError("Kernel and stride must be provided in the configuration or command line arguments. Please ensure they are set correctly.")

    # initialize the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Check if dataset paths are provided
    if not args.dataset_atomic or not args.dataset_crystal:
        raise ValueError("Both dataset_atomic and dataset_crystal must be provided.")
    # Check if dataset paths exist
    if not os.path.exists(args.dataset_atomic):
        raise FileNotFoundError(f"Atomic dataset file not found: {args.dataset_atomic}")
    if not os.path.exists(args.dataset_crystal):
        raise FileNotFoundError(f"Crystal dataset file not found: {args.dataset_crystal}")

    # Load dataset paths
    X = np.load(args.dataset_atomic, allow_pickle=True)
    Y = np.load(args.dataset_crystal, allow_pickle=True)
    if X.shape[0] != Y.shape[0]:
        raise ValueError("The number of samples in atomic and crystal datasets must match.")
    
    logging.info("Loaded datasets with %d samples.", X.shape[0])
    logging.info("Atomic dataset shape: %s, Crystal dataset shape: %s", X.shape, Y.shape)
   
    # Setup scaler for Y
    scaler_Y = StandardScaler()
    Y_scaled = Y.copy()
    Y_scaled[:,:6] = scaler_Y.fit_transform(Y[:,:6])
    # Set X and Y to torch tensors
    Y = Y_scaled
    
    dataset = CrystalDataset(X, Y)
    logging.info("Dataset created with %d samples.", len(dataset))
    # Split the dataset into training, validation, and test sets
    datasplits = split_data(dataset, args)

    logging.info("Train:", datasplits['train'].dataset.X.shape, datasplits['train'].dataset.Y.shape,)
    logging.info("Validation:", datasplits['val'].dataset.X.shape, datasplits['val'].dataset.Y.shape)
    logging.info("Test:", datasplits['test'].dataset.X.shape, datasplits['test'].dataset.Y.shape)

    # Initialize the VAE
    feature_dim = datasplits['train'].dataset.X.shape[2]
    wyckoff_dim = datasplits['train'].dataset.X.shape[1]
    crystal_dim = datasplits['train'].dataset.Y.shape[1]
    logging.info("Feature dimension: %d, Wyckoff dimension: %d, Crystal dimension: %d", feature_dim, wyckoff_dim, crystal_dim)

    vae = VAE(
        feature_dim,
        wyckoff_dim,
        crystal_dim,
        space_group_dim=230,  # Assuming 230 space groups
        lattice_dim=6,  # Assuming 6 lattice parameters
        stride=args.stride,
        kernel=args.kernel,
        latent_dim=args.latent_dim,
        max_filter=args.max_filter,
        verbose= args.verbose if hasattr(args, 'verbose') else False
    )

    vae.to(device)

    optimizer = Adam(vae.parameters(), lr=args.learning_rate)

    coeffs = args.loss_weights 
    default_coeffs = {'kl': 1.0, # 1
          'element':2000.0, #2000
          'wyckoff_letter': 1.0,
        'wyckoff_multiplier': 1.0,
          'frac_coords': 2.0, # 1
          'space_group':10.0,
          'lattice':3.0,
          'disordered_site':0.1}
    model_coeffs = {'feature_dim': feature_dim,
                    'wyckoff_dim': wyckoff_dim,
                    'crystal_dim': crystal_dim,
                    'latent_dim': args.latent_dim,
                    'kernel': args.kernel,
                    'stride': args.stride,
                    'max_filter': args.max_filter}

    
    if coeffs is None or coeffs.keys() != default_coeffs.keys():
        logging.warning("Loss weights not provided or incomplete. Using default weights.")
        coeffs = default_coeffs
    logging.info("Loss coefficients: %s", coeffs)

    # Save loss coefficients
    with open(f'{args.output_dir}/coeffs.json', 'w') as f:
        json.dump(coeffs, f)
    

    # Initialize the train and validation loss
    train_loss_dict = {}
    train_loss_dict['total'] = []
    train_loss_dict['epoch'] = []

    val_loss_dict = {}
    val_loss_dict['total'] = []
    val_loss_dict['epoch'] = []

    best_val_loss = np.inf

    # Setup the batch size and shuffle the datasets
    for key, items in datasplits.items():
        datasplits[key] = DataLoader(items, batch_size=20, shuffle=True)

    # Training loop
    for epoch in tqdm(range(args.epochs)):
        vae.train()
        # Initialize the train loss
        train_loss = 0.0
        train_running_loss = {}

        # Loop through the train loader
        for x_batch, y_batch in datasplits['train']:  # Use both datasets if required
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

        logging.info(f"Epoch {epoch + 1}/{args.epochs}: Train Loss: {train_loss / len(datasplits['train'])}")
        
        # Save the train loss for each component of the loss as a mean over the batches
        for key in train_running_loss:
            if key not in train_loss_dict:
                train_loss_dict[key] = [train_running_loss[key].item() / len(datasplits['train'])]
            else:
                train_loss_dict[key].append(train_running_loss[key].item() / len(datasplits['train']))
        train_loss_dict['epoch'].append(epoch)
        
        # Validate and save model for each log step
        if (epoch + 1) % args.log_step == 0:
            vae.eval()
            val_loss = 0.0
            val_running_loss = {}
            with torch.no_grad():
                for x_val, y_val in datasplits['val']:
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
            logging.info('---------------------------------')
            logging.info(f"Validation Loss:")
            
            for key in val_running_loss:
                # Print all losses
                logging.info(f"{key}: {val_running_loss[key].item()/len(datasplits['val'])}")
                # Save the validation loss
                if key not in val_loss_dict:
                    val_loss_dict[key] = [val_running_loss[key].item()/len(datasplits['val'])]
                else:
                    val_loss_dict[key].append(val_running_loss[key].item()/len(datasplits['val']))
            val_loss_dict['epoch'].append(epoch)


            # Save the model if the validation loss is the best
            if val_running_loss['total'] < best_val_loss:
                best_val_loss = val_running_loss['total']
                torch.save({"model": vae.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "coeffs": coeffs,
                            "epoch": epoch,
                            "total loss": best_val_loss,
                            "val_loss": val_loss_dict,
                            "train_loss": train_loss_dict,
                            "scaler_Y": scaler_Y,
                            "model_coeffs": model_coeffs
                            },      
                            f'{args.output_dir}/best_vae_model.pth')
                print('Model saved')

    ###### Plot the losses ##############################################
    torch.save({"model": vae.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "coeffs": coeffs,
                            "epoch": epoch,
                            "total loss": best_val_loss,
                            "val_loss": val_loss_dict,
                            "train_loss": train_loss_dict,
                            "scaler_Y": scaler_Y,
                            "model_coeffs": model_coeffs
                            },      
                        f'{args.output_dir}/exit_vae_model.pth')
    
    ## Plot all the losses as a function of the epoch
    fig, ax = plt.subplots(1,2,figsize=(20,10))
    font_size = 20
    Nbins = 20

    for key in train_loss_dict:
        if key == 'epoch':
            continue
        ax[0].plot(train_loss_dict[key],label=key)
        val_epoch  = np.arange(0,len(val_loss_dict[key])*args.log_step,args.log_step)
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
    plt.savefig(f'{args.output_dir}/Losses.png')

    # Save the losses
    train_loss_df = pd.DataFrame(train_loss_dict)
    val_loss_df = pd.DataFrame(val_loss_dict)
    train_loss_df.to_csv(f'{args.output_dir}/train_loss.csv')
    val_loss_df.to_csv(f'{args.output_dir}/val_loss.csv')

if __name__ == "__main__":
    main()
