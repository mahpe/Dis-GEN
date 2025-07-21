import torch
import torch.nn.functional as F

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
    decoded_disordered_site = model_output['decoded_disordered_site']

    if verbose:
        print('Original:',x.shape)
        print('Reconstructed:',decoded_element.shape,decoded_wyckoff_multiplier.shape,decoded_frac_coords.shape,decoded_wyckoff_letter.shape,decoded_disordered_site.shape)
        print('Z mean:',z_mean.shape)
        print('Z log var:',z_log_var.shape)
        print('Coeffs:',coeffs)
        print('---------------------------------')
    
    #### KL Divergence loss ####
    #kl_loss_i = 0.5*( -torch.log(torch.square(z_log_var)) - 1 + torch.square(z_log_var) + torch.square(z_mean) )
    #kl_loss = torch.mean(torch.sum(kl_loss_i, dim=1)) # sum over the latent dimensions and mean over the batch
    
    kl_loss_i = (1 + z_log_var - torch.square(z_mean) - torch.exp(z_log_var))
    kl_loss = torch.mean( -0.5 * (1/z_mean.shape[1]) * torch.sum(kl_loss_i, dim=1) ) # sum over the latent dimensions and mean over the batch
    losses['kl'] = kl_loss
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
    disordered_site_loss_i = []
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
        wyckoff_multiplier_loss_i.append(F.cross_entropy(decoded_wyckoff_multiplier[:,i,:], x[:,i,101:-31],reduction='none',))
        
        # Disordered site loss. Cross entropy loss
        #disordered_site_loss_i.append(torch.square(decoded_disordered_site[:,i,0] - x[:,i,-31]) )
        #print(x[:,i,-31][:,None], decoded_disordered_site[:,i,:])
        disordered_site_loss_i.append(F.binary_cross_entropy_with_logits(decoded_disordered_site[:,i,0], x[:,i,-31],reduction='none'))
#        print(decoded_disordered_site[:,i,:].t().shape,x[:,i,-31][None,:].shape)
        #print(F.binary_cross_entropy_with_logits(decoded_disordered_site[:,i,0], x[:,i,-31],reduction='none'))

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

    # Disordered site loss. Cross entropy loss
    disordered_site_loss_i = torch.stack(disordered_site_loss_i)
    disordered_site_loss = torch.mean(torch.sum(disordered_site_loss_i,dim=0)) # sum over the wyckoff sites and mean over the batch
    losses['disordered_site'] = disordered_site_loss
    if verbose:
        print('Disordered Site Loss i:',disordered_site_loss_i.shape)
        print('Disordered Site Loss:',disordered_site_loss.shape)
    
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