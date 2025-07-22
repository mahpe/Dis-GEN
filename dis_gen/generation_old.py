import numpy as np
import torch
import torch.nn.functional as F
import numpy as np
from pymatgen.core import Lattice, Structure
from pymatgen.symmetry.groups import SpaceGroup
from pymatgen.symmetry.structure import SymmetrizedStructure
from pymatgen.symmetry.analyzer import SpacegroupOperations
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.coord import in_coord_list
from ase.data import chemical_symbols


def decode_samples_xy(X,Y,scaler_Y=None, device='cpu'):
    """
    Decode samples from representation atomic and crystal features.
    
    Args:
    X (np.ndarray or torch.Tensor): samples from the atomic representation.
    Y (np.ndarray or torch.Tensor): samples from the crystal representation.
    scaler_Y (StandardScaler): Scaler used to transform the crystal features., None is default and no scaling is applied.
    device (str): Device to run the decoding on ('cpu' or 'cuda').
    Returns:
    dict: A dictionary containing decoded crystal features including lattice parameters, space group, and atomic features
    """
    # To numpy if input is torch.Tensor
    
    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()
    elif isinstance(X, list):
        X = np.array(X, dtype=np.float32)
    if isinstance(Y, torch.Tensor):
        Y = Y.cpu().numpy()
    elif isinstance(Y, list):
        Y = np.array(Y, dtype=np.float32)
    

    lattice = Y[:, :6]  # Lattice parameters
    sg = Y[:, 6:]

    if scaler_Y is not None:
        lattice = scaler_Y.inverse_transform(lattice)

    element = X[:,:,:101]
    frac_coords = X[:,:,-30:-27]
    wyckoff_multiplier = X[:,:,101:-31]
    wyckoff_letter = X[:,:,-27:]
    disordered_site = X[:,:,-31]

    save_dict = {
        'abc': np.round(lattice[:, :3], 1),
        'angles': np.round(lattice[:, 3:6], 1),
        'spacegroup': np.argmax(sg, axis=1)+1, # Spacegroup is 1-indexed in pymatgen
        'disordered_site': disordered_site,
        'element': element,
        'wyckoff_letter': wyckoff_letter,
        'wyckoff_mult': wyckoff_multiplier,
        'frac_coords': frac_coords
    }
    return save_dict



def decode_samples(z_samples,vae_eval,scaler_Y, device='cpu'):
    """
    Decode samples from the latent space using a trained VAE model.

    Args:
    z_samples (np.ndarray or torch.Tensor): Samples from the latent space to decode.
    vae_eval (VAE): The trained VAE model for decoding.
    scaler_Y (StandardScaler): Scaler used to transform the crystal features.
    device (str): Device to run the decoding on ('cpu' or 'cuda').
    Returns:
    dict: A dictionary containing decoded crystal features including lattice parameters, space group, and atomic features.
    """

    if isinstance(z_samples, torch.Tensor):
        z_samples = z_samples.to(device)
    elif isinstance(z_samples, list):
        z_samples = torch.tensor(z_samples, dtype=torch.float32).to(device)
    elif isinstance(z_samples, np.ndarray):
        z_samples = torch.tensor(z_samples, dtype=torch.float32).to(device)

    decode = vae_eval.decode(z_samples)
    element,wyckoff_multiplier,frac_coords,wyckoff_letter, sg, lattice,disordered_site = decode
    lattice = scaler_Y.inverse_transform(lattice.detach().numpy())
    save_dict = {'abc':np.round(lattice[:, :3],1),'angles':np.round(lattice[:, 3:6],1),'spacegroup':np.argmax(sg.detach().numpy(),axis=1)+1,
                'disordered_site': F.sigmoid(torch.tensor(disordered_site)).cpu().detach().numpy(),
                'element':element.detach().numpy(),'wyckoff_letter':wyckoff_letter.detach().numpy(),
                'wyckoff_mult':wyckoff_multiplier.detach().numpy(),'frac_coords':frac_coords.detach().numpy()}
    return save_dict


def wyckoff_expand(w_site):
    """"
    Expand Wyckoff sites into their respective indices and letters."
    Args:
        w_site (list): List of Wyckoff sites in the form of "1a", "2b", etc.
    Returns:
        tuple: Two lists - one with the expanded indices and another with the corresponding letters.
    """
    # Extract the multipliers and letters from the Wyckoff sites
    # Example: "2a" -> 2, "1b" -> 1
    # Example: "2a" -> "a"
    # Example: "1b" -> "b"
    mult_indexes = []
    letters = []
    index = 0
    
    for item in w_site:
        count = int(item[:-1])  # Extract number -> is this covering double digits
        label = item[-1]       # Extract label
        mult_indexes.extend([index] * count)
        letters.extend([label] * count)
        index += count
    
    return np.array(mult_indexes, dtype=np.int32), letters

def symm_structure_maker(structure, sg_number, equivalent_positions, wyckoff_letters,sym_tolerance):
    """
    Create a symmetrized structure from a given structure and space group number.
    Args:
        structure (Structure): The input structure to be symmetrized.
        sg_number (int): The international number of the space group.
        equivalent_positions (list): List of equivalent positions for the Wyckoff sites.
        wyckoff_letters (list): List of Wyckoff letters corresponding to the equivalent positions.
        sym_tolerance (float): Tolerance for symmetry operations.
    Returns:
        SymmetrizedStructure: A symmetrized structure object containing the full structure and symmetry operations.
    """
    sym_ops = SpaceGroup.from_int_number(sg_number).symmetry_ops   # Obtains a SpaceGroup from its international number

    full_coords = []
    full_species = []

    # Loop through the sites and apply symmetry operations    
    for site in structure.sites:
        for sym_op in sym_ops:
            transformed_coords = sym_op.operate(site.frac_coords) % 1  # bring back to unit cell
            species = site.species
            full_coords.append(transformed_coords)
            full_species.append(species)
    
    # Remove duplicates
    unique_coords = []
    unique_species = []
    tolerance = sym_tolerance
    for i, coord in enumerate(full_coords):
        if not in_coord_list(unique_coords, coord, atol=tolerance):
            unique_coords.append(coord)
            unique_species.append(full_species[i])

    full_structure = Structure(structure.lattice, unique_species, unique_coords)

    #spg = SpacegroupAnalyzer(full_structure,symprec=tolerance)
    #conventional_structure = spg.get_conventional_standard_structure()
    #symmetrized_structure = spg.get_symmetrized_structure()
    #print(spg.get_space_group_number())


    symm_struct = SymmetrizedStructure(
        structure=full_structure,
        spacegroup=SpacegroupOperations(SpaceGroup.from_int_number(sg_number).full_symbol, sg_number, sym_ops),
        equivalent_positions=equivalent_positions,
        wyckoff_letters=wyckoff_letters
    )
    
    #if not structure_validity(symm_struct):
    #    raise ValueError("Invalid structure")
    
    #if SpacegroupAnalyzer(symm_struct,symprec=tolerance).get_space_group_number() != sg_number:
    #  raise ValueError(f"Spacegroup mismatch: {sg_number} vs {SpacegroupAnalyzer(symm_struct).get_space_group_number()}")
    
    #print('Real:',sg_number,'Full:',SpacegroupAnalyzer(full_structure).get_space_group_number(),'Sym:',SpacegroupAnalyzer(symm_struct).get_space_group_number())

    #spg = SpacegroupAnalyzer(symm_struct,symprec=tolerance)
    #conventional_structure = spg.get_conventional_standard_structure()
    #symmetrized_structure = spg.get_symmetrized_structure()

    #return symmetrized_structure
    return symm_struct
    #return full_structure

def process_data_pkl(data_pkl, max_iter = None,element_acc = 0.01,disorder_acc = 0.1,sym_tolerance = 0.001):
    """
    Process the data from a dictionary containing crystal structure information and return a list of symmetrized structures.
    Args:
        data_pkl (dict): Dictionary containing crystal structure data with keys 'abc', 'angles', 'spacegroup', 'element', 'wyckoff_letter', 'wyckoff_mult', 'frac_coords', and 'disordered_site'.
        max_iter (int, optional): Maximum number of structures to process. If None, all structures are processed.
        element_acc (float, optional): Minimum fraction for an element to be considered present.
        disorder_acc (float, optional): Threshold for determining if a site is disordered.
        sym_tolernace (float, optional): Tolerance for symmetry operations.
    Returns:
        list: A list of symmetrized structures.
    """

    symm_structs = []
    not_symm_structs = []
    failed_count = 0  # Counter for failed items
    
    for i in range(len(data_pkl['abc'])):
        try:
            # Define the lattice parameters
            lattice = Lattice.from_parameters(
                a=data_pkl['abc'][i][0],
                b=data_pkl['abc'][i][1],
                c=data_pkl['abc'][i][2],
                alpha=data_pkl['angles'][i][0],
                beta=data_pkl['angles'][i][1],
                gamma=data_pkl['angles'][i][2]
            )

            # Identify if a Wyckoff site is empty
            index_max = np.argmax(data_pkl['wyckoff_letter'][i], axis=1)
            index = index_max != 0

            # Define if the site is disordered
            disordered_site = data_pkl['disordered_site'][i][index]
            disordered_site = disordered_site>disorder_acc
            disordered_site[~disordered_site] = 0
            disordered_site = disordered_site<disorder_acc
            disordered_site[~disordered_site] = 1
            disordered_site = disordered_site

            # Loop trough all wyckoff sites and define the element
            element_comb = []
            for site, a_list in enumerate(data_pkl['element'][i][index]):
                
                if disordered_site[site] == 1: # Disordered
                    element_index = np.where(a_list > element_acc)[0] + 1
                    disordered = True
                else: # Ordered
                    element_index = np.argmax(a_list) + 1
                    disordered = False                    

                if disordered:
                    disordered_element = {chemical_symbols[elem]: round(float(data_pkl['element'][i][site][elem - 1]),2) for elem in element_index}
                    element_comb.append(disordered_element)
                else:
                    element_comb.append({chemical_symbols[element_index]: 1.0})
                # Remove Md from element_comb
                #element_comb = [d for d in element_comb if 'Md' not in d]
            #print(element_comb)
            # Define the Fractional coordinates
            frac_coords = data_pkl['frac_coords'][i][index]

            # Create the structure            
            structure = Structure(lattice, element_comb, frac_coords)
            #print('Structure:',structure)
            not_symm_structs.append(structure)
            
            # Define the spacegroup
            spacegroup_int = data_pkl['spacegroup'][i]
            #print('Spacegroup:',spacegroup_int)

            # Define the Wyckoff sites letters and multipliers            
            w_letter = np.array([chr(ord('a') + l - 1) for l in np.argmax(data_pkl['wyckoff_letter'][i][index], axis=1) if l != 0])
            w_multiplier = np.array([m for m in np.argmax(data_pkl['wyckoff_mult'][i][index], axis=1) if m != 0])
            w_site = [str(w) + l for w, l in zip(w_multiplier, w_letter)]

            # Apply the Wyckoff letters and multipliers
            mult_indexes, letters = wyckoff_expand(w_site)

            # Create the structure with symmetry
            symm_struct = symm_structure_maker(structure, spacegroup_int, mult_indexes, letters,sym_tolerance)
            
            symm_structs.append(symm_struct)
            #print(f"Successfully processed index {i}. Total symm_structs: {len(symm_structs)}")
        except Exception as e:
            print(f"Error processing index {i}: {e}")
            failed_count += 1  # Increment the failed counter
            
            
            # print(element_comb)
        if i % 1000 == 0:
            print(f"Processed index {i} out of {len(data_pkl['abc'])}")
            print(f"Total failed: {failed_count} out of {len(data_pkl['abc'])}")
        if max_iter is not None and i == max_iter:
            break
    return symm_structs
    #return not_symm_structs