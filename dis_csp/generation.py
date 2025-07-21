
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core import Lattice, Structure, Molecule
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core import PeriodicSite

from tqdm import tqdm
from pymatgen.core import Composition, Structure
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
from dis_csp.csp_filter import structure_validity,oxidation_state_validity,Symmetry_matching_accuracy
from collections import defaultdict
import pyxtal as px
from pyxtal.tolerance import Tol_matrix
from pyxtal.symmetry import search_cloest_wp, Group
from pymatgen.io.cif import CifParser
from io import StringIO

SG_SYM = {spacegroup: Group(spacegroup) for spacegroup in range(1, 231)}
SG_TO_WP_TO_SITE_SYMM = dict()
for spacegroup in range(1, 231):
    SG_TO_WP_TO_SITE_SYMM[spacegroup] = dict()
    for wp in SG_SYM[spacegroup].Wyckoff_positions:
        wp_site = str(wp.multiplicity)+wp.letter
        wp.get_site_symmetry()
        SG_TO_WP_TO_SITE_SYMM[spacegroup][wp_site] = wp
#       SG_TO_WP_TO_SITE_SYMM[spacegroup][wp_site][wp] = wp.get_site_symmetry_object().to_one_hot()



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


def align_fractional_coordinates(fractional_coord,wp_site,spg,spg_group):
    """
    Align the fractional coordinates to the Wyckoff position.
    
    Args:
        fractional_coord (np.ndarray): The fractional coordinates to align.
        wp_site (str): The Wyckoff position site (e.g., '2a', '3b').
        spg (int): The space group number.
        spg_group (pyxtal.symmetry.Group): The space group object.
    
    Returns:
        np.ndarray: The aligned fractional coordinates.
        pyxtal.symmetry.WyckoffPosition: The Wyckoff position object.
        int: The orbit index of the Wyckoff position.
    """
    wp = SG_TO_WP_TO_SITE_SYMM[spg][wp_site]

    closes =[]
    for orbit_index in range(len(wp.ops)):
        close = search_cloest_wp(spg_group, wp, wp.ops[orbit_index], fractional_coord)
        closes.append((close, wp, orbit_index, np.linalg.norm(np.minimum((close - fractional_coord)%1., (fractional_coord - close)%1.)) ))

    # Get the closest fractional coordinates
    closest = sorted(closes, key=lambda x: x[-1])[0]
    return closest[0], closest[1], closest[2]  # Return the closest fractional coordinates, Wyckoff position, and orbit index 


def generate_wyckoffgene(data_pkl, max_iter = None,element_acc = 0.01,disorder_acc = 0.1,shift_frac_coord=True,verbose = False):
    """
    Process the data from a dictionary containing crystal structure information and return a list of symmetrized structures.
    Args:
        data_pkl (dict): Dictionary containing crystal structure data with keys 'abc', 'angles', 'spacegroup', 'element', 'wyckoff_letter', 'wyckoff_mult', 'frac_coords', and 'disordered_site'.
        max_iter (int, optional): Maximum number of structures to process. If None, all structures are processed.
        element_acc (float, optional): Minimum fraction for an element to be considered present.
        disorder_acc (float, optional): Threshold for determining if a site is disordered.
        shift_frac_coord (bool, optional): If True, shift fractional coordinates to align with Wyckoff positions.
        verbose (bool, optional): If True, print additional information during processing.
    Returns:
        list: A list of wyckoffgene dictionaries, each containing information about a crystal structure.
        int: Number of symmetry matching accuracy failures.
        int: Number of failed items due to invalid lattice parameters or other issues.
    """

    wyckoffgenes_list = [] 
    failed_count = 0  # Counter for failed items
    SMA_count = 0  # Counter for symmetry matching accuracy failures
    for i in tqdm(range(len(data_pkl['abc']))):
        # # Create the structure            
        spacegroup_int = data_pkl['spacegroup'][i]
        spacegroup_group = SG_SYM[spacegroup_int] if spacegroup_int in SG_SYM else Group(spacegroup_int)     
        
        # Define the lattice parameters
        lattice = Lattice.from_parameters(
            a=data_pkl['abc'][i][0],
            b=data_pkl['abc'][i][1],
            c=data_pkl['abc'][i][2],
            alpha=data_pkl['angles'][i][0],
            beta=data_pkl['angles'][i][1],
            gamma=data_pkl['angles'][i][2]
        )
        # Check if the lattice is valid
        if any(np.array(lattice.abc) == 0):
            if verbose:
                print(f"Invalid lattice parameters for index {i}: {lattice.abc}")
            failed_count += 1
            continue
        # create lattic object
        # Identify if a Wyckoff site is empty
        index_max = np.argmax(data_pkl['wyckoff_letter'][i], axis=1)
        index = index_max != 0

        # Define if the site is disordered
        disordered_site = data_pkl['disordered_site'][i][index]
        disordered_site = disordered_site>disorder_acc

        # Loop trough all wyckoff sites and define the element
        index_array = np.arange(0,len(index))[index]
        element_comb = []
        for site, a_list in enumerate(data_pkl['element'][i][index]):

            if disordered_site[site]: # Disordered
                element_index = np.where(a_list > element_acc)[0] + 1
                disordered = True
            else: # Ordered
                element_index = np.argmax(a_list) + 1
                disordered = False                    
            
            if disordered:
                disordered_element = {chemical_symbols[elem]: round(float(data_pkl['element'][i][site][elem - 1]),2) for elem in element_index if chemical_symbols[elem] !='Md'} # Md is vacancies
                if len(disordered_element) == 0:  # If no elements are present, skip this site
                    index[index_array[site]] = False  # Mark the site as not present
                else:
                    element_comb.append(disordered_element)
            else:
                if chemical_symbols[element_index] == 'Md':  # If the element is Md, it is a vacancy
                    index[index_array[site]] = False  # Mark the site as not present
                else:
                    element_comb.append({chemical_symbols[element_index]: 1.0})
        
        # Define the Fractional coordinates
        frac_coords = data_pkl['frac_coords'][i][index]
        
        # Define the Wyckoff sites letters and multipliers            
        w_letter = np.array([chr(ord('a') + l - 1) for l in np.argmax(data_pkl['wyckoff_letter'][i][index], axis=1) if l != 0])
        w_multiplier = np.array([m for m in np.argmax(data_pkl['wyckoff_mult'][i][index], axis=1) if m != 0])
        w_site = [str(w) + l for w, l in zip(w_multiplier, w_letter)]

        # Check if the length of Wyckoff letters and multipliers match. A mismatch can happen if either the letter or multiplier is zero.
        if len(w_letter) != len(w_multiplier):
            if verbose:
                print(index)
                print(w_letter, w_multiplier,w_site)
                print(f"Mismatch in length of Wyckoff letters and multipliers for index {i}.")
            failed_count += 1
            continue
        
        # Check if the symmetry matching is correct
        symm_acc = Symmetry_matching_accuracy(w_letter, w_multiplier, spacegroup_group)
        if not symm_acc:
            if verbose:
                print(f"Symmetry accuracy failed for index {i} with spacegroup {spacegroup_int} and Wyckoff sites {w_site}")
            SMA_count += 1
            continue
       
        # Shift fractional coordinates if needed 
        if shift_frac_coord:
            coord_list = []
            for wp_site, coord in zip(w_site, frac_coords):

                # Extra check for symmetry accuracy
                if wp_site not in SG_TO_WP_TO_SITE_SYMM[spacegroup_int].keys():
                    SMA_count += 1
                    symm_acc = False
                    break 

                # Align the fractional coordinates to the Wyckoff position
                coord, _, _ = align_fractional_coordinates(coord, wp_site, spacegroup_int, spacegroup_group)
                coord_list.append(coord)
            
            frac_coords = coord_list
            
            if not symm_acc:
                if verbose:
                    print(f"Wyckoff site {wp_site} not found in spacegroup {spacegroup_int}.")
                continue 

            
        
        # Get the wyckoffgene 
        element_wyckoff = defaultdict(list)
        element_counts = defaultdict(int)
        elements_occ = defaultdict(lambda: defaultdict(list))
        elements_frac_coord = defaultdict(lambda: defaultdict(list))

        for el_dict, wyck, fr in zip(element_comb, w_site, frac_coords):
            for el, occ in el_dict.items():
                element_wyckoff[el].append(wyck)
                element_counts[el] += int(round(int(''.join(filter(str.isdigit, wyck)))))  # assume multiplicity from Wyckoff symbol
                elements_occ[el][wyck].append(occ)  # accumulate occupancy for each element and wyckoff site
                elements_frac_coord[el][wyck].append(fr)  # accumulate fractional coordinates for each elemen        

        # Step 2: Build wyckoffgene
        wyckoffgene = {
            'spacegroup': spacegroup_int,
            'species': list(element_wyckoff.keys()),
            'sites': list(element_wyckoff.values()),
            'numIons': list(element_counts.values()),
            'lattice':px.lattice.Lattice(ltype=spacegroup_group.lattice_type,
                        volume=lattice.volume,
                        matrix=lattice.matrix
                        ),
            'occupancy': elements_occ,  # Assuming full occupancy for simplicity    
            'frac_coord': elements_frac_coord,
            'wyckoff_sites_list':w_site,
            'frac_coord_list': frac_coords,
            'structure_pymatgen': Structure(lattice, element_comb, frac_coords),
            'spacegroup_group':spacegroup_group,
        }

        wyckoffgenes_list.append(wyckoffgene)
        if max_iter is not None and i >= max_iter:
            break
    return wyckoffgenes_list, SMA_count, failed_count

def split_wyckoff(site):
    """
    Split the Wyckoff site into multiplicity and letter.
    
    Args:
        site (str): The Wyckoff site string (e.g., '2a', '3b').
    
    Returns:
        tuple: A tuple containing the multiplicity (int) and the letter (str).
    """
    for i, c in enumerate(site):
        if c.isalpha():
            return int(site[:i]), site[i:]
        

def get_cif_lines(wyckoffgene_dict):
    """
    Generate CIF lines from a wyckoffgene dictionary.

    Args:
        wyckoffgene_dict (dict): Dictionary containing crystal structure information with keys 'spacegroup_group', 'lattice', 'sites', 'species', 'frac_coord', and 'occupancy'.
    
    Returns:
        str: A string containing the CIF lines.
    """

    # Initiate the cif lines
    lines = ""
    # Symmetry information
    spg_group = wyckoffgene_dict['spacegroup_group']
    l_type =spg_group.lattice_type
    number = spg_group.number
    G1 = spg_group[0]
    symbol =spg_group.symbol #if G1.is_standard_setting()

    lines += "data_\n"

    lines += f"\n_symmetry_space_group_name_H-M '{symbol:s}'\n"
    lines += f"_symmetry_Int_Tables_number      {number:>15d}\n"
    lines += f"_symmetry_cell_setting           {l_type:>15s}\n"

    a, b, c, alpha, beta, gamma = wyckoffgene_dict['lattice'].get_para(degree=True)
    lines += f"_cell_length_a        {a}\n"
    lines += f"_cell_length_b        {b}\n"
    lines += f"_cell_length_c        {c}\n"
    lines += f"_cell_angle_alpha     {alpha}\n"
    lines += f"_cell_angle_beta      {beta}\n"
    lines += f"_cell_angle_gamma     {gamma}\n"
    lines += f"_cell_volume          {wyckoffgene_dict['lattice'].volume}\n"
    lines += "\nloop_\n"
    lines += " _symmetry_equiv_pos_site_id\n"
    lines += " _symmetry_equiv_pos_as_xyz\n"
    for i, op in enumerate(G1):
        lines += f"{i + 1:d} '{op.as_xyz_str():s}'\n"

    lines += "\nloop_\n"
    lines += " _atom_site_label\n"
    lines += " _atom_site_type_symbol\n"
    lines += " _atom_site_symmetry_multiplicity\n"
    lines += " _atom_site_Wyckoff_symbol\n" # ICSD style
    lines += " _atom_site_fract_x\n"
    lines += " _atom_site_fract_y\n"
    lines += " _atom_site_fract_z\n"
    lines += " _atom_site_occupancy\n"
    #print(wyckoffgene_dict['sites'])
    count= {}
    for i in range(len(wyckoffgene_dict['sites'])):
        wp_list = wyckoffgene_dict['sites'][i]
        for wp_site in wp_list:
            # Split the Wyckoff site into multiplicity and letter
            mul, letter = split_wyckoff(wp_site)
            
            specie = wyckoffgene_dict["species"][i]
            coord = wyckoffgene_dict['frac_coord'][specie][wp_site]
            occ = wyckoffgene_dict['occupancy'][specie][wp_site]

            if len(coord) >1: # if there are 2 frac coordinates at the same site for the same element. Exp. S is at 6c, twich
                if specie not in count.keys():
                    count[specie] = {wp_site: 0}
                else:
                    if wp_site not in count[specie].keys():
                        count[specie][wp_site] = 0

                coord_i = coord[count[specie][wp_site]]
                occ_i = occ[count[specie][wp_site]]
                count[specie][wp_site] += 1
            else:
                coord_i = coord[0]
                occ_i = occ[0]


            lines += f"{specie:6s} {specie:6s} {mul:3d} {letter:s}"
            lines += "{:12.3f}{:12.3f}{:12.3f}".format(*coord_i)
            #lines += f" {coord_i[0]} {coord_i[1]} {coord_i[2]}"
            lines += f" {occ_i} \n"

            
    lines += "#END\n\n"
    return lines


def generate_cif_files(all_wyckoffgenes_list,maxiter=None, validity_check=True,charge_neutral=True,
                       two_oxidation_state=False,validity_primitive=False,symmetry_analyzer=False, verbose=False):
    """
    Generate CIF files from a list of WyckoffGene dictionaries.
    Parameters:
    all_wyckoffgenes_list (list): List of WyckoffGene dictionaries.
    maxiter (int): Maximum number of structures to process. If None, process all.
    validity_check (bool): If True, check the validity of the structures.
    charge_neutral (bool): If True, check the oxidation state validity.
    two_oxidation_state (bool): If True, allow two oxidation states.
    validity_primitive (bool): If True, use the primitive structure for validity checks.
    symmetry_analyzer (bool): If True, use pymatgen's symmetry analyzer.
    verbose (bool): If True, print additional information.
    Returns:
    all_cif_lines (list): List of CIF lines for valid structures.
    cif_lines_wyckoffgene (list): List of CIF lines with corresponding WyckoffGene dictionaries.
    """

    all_cif_lines = []
    cif_lines_wyckoffgene = []
    failed_count = 0
    sym_count = 0
    validity_count = 0
    oxidation_state_count = 0
    validity_count_list = []

    struc_valid = True # If they are not chossen as input, they are valid by default
    oxidation_valid = True # If they are not chossen as input, they are valid by default
    if maxiter is not None:
        print(f"Processing {maxiter} structures out of {len(all_wyckoffgenes_list)}")

    for i in tqdm(range(len(all_wyckoffgenes_list))):
        if maxiter is not None:
            if i >= maxiter:
                break
        wyckoffgene = all_wyckoffgenes_list[i]

        cif_lines = get_cif_lines(wyckoffgene)
        

        try:
            # primitve cell
            if validity_primitive:
                structure = wyckoffgene['structure_pymatgen']
            # Unit cell
            else:
                # Found that pymatgen had some unknown issues readin some cif files, while ase had no errors. Since the efficiency only increased by 10% we will do it with ase
                #structure = Structure.from_str(cif_lines,fmt='cif')
                cif_file = StringIO(cif_lines)
                atoms = read(cif_file, format="cif")
                structure = AseAtomsAdaptor.get_structure(atoms)
        except:
            failed_count += 1
            continue

        if symmetry_analyzer:
            try:
                sga = SpacegroupAnalyzer(structure,symprec=0.1)
                refined_struc = sga.get_refined_structure()
                sga = SpacegroupAnalyzer(refined_struc,symprec=0.01)
                structure = sga.get_symmetrized_structure()
            except:
                sym_count +=1
                continue
        
        if validity_check:
            struc_valid =structure_validity(structure)
            if not struc_valid:
                validity_count += 1
        
        if charge_neutral:   
            oxidation_valid = oxidation_state_validity(structure,two_oxidation_state=two_oxidation_state,verbose=verbose)
            if not oxidation_valid:
                oxidation_state_count += 1
    
        # if one of them is not valid, skip the structures
        if not struc_valid:
            continue 
        if not oxidation_valid:
            continue

        # Add the structure to the list
        all_cif_lines.append(cif_lines)
        cif_lines_wyckoffgene.append(wyckoffgene)

    if maxiter is not None:
        print(f'Failed to parse {failed_count} structures out of {maxiter} Procentage: {100-failed_count/maxiter*100:.2f}%')
        if symmetry_analyzer:
            print(f'Pymatgen Symmetry failed for {sym_count} structures out of {maxiter} Procentage: {100-sym_count/maxiter*100:.2f}%')
        if validity_check:
            print(f'Validity check failed for {validity_count} structures out of {maxiter} Procentage: {100-validity_count/maxiter*100:.2f}%')
        if charge_neutral:
            print(f'Oxidation state check failed for {oxidation_state_count} structures out of {maxiter} Procentage: {100-oxidation_state_count/maxiter*100:.2f}%')
        print(f'Total valid structures: {len(all_cif_lines)} out of {maxiter} Procentage: {len(all_cif_lines)/maxiter*100:.2f}%')
    else:
        print(f"Failed to parse {failed_count} structures out of {len(all_wyckoffgenes_list)} Procentage: {100-failed_count/len(all_wyckoffgenes_list)*100:.2f}%")
        if symmetry_analyzer:
            print(f"Pymatgen Symmetry failed for {sym_count} structures out of {len(all_wyckoffgenes_list)} Procentage: {100-sym_count/len(all_wyckoffgenes_list)*100:.2f}%")
        if validity_check:
            print(f"Validity check failed for {validity_count} structures out of {len(all_wyckoffgenes_list)} Procentage: {100-validity_count/len(all_wyckoffgenes_list)*100:.2f}%")
        if charge_neutral:
            print(f"Oxidation state check failed for {oxidation_state_count} structures out of {len(all_wyckoffgenes_list)} Procentage: {oxidation_state_count/len(all_wyckoffgenes_list)*100:.2f}%")
        print(f"Total valid structures: {len(all_cif_lines)} out of {len(all_wyckoffgenes_list)} Procentage: {len(all_cif_lines)/len(all_wyckoffgenes_list)*100:.2f}%")    

    return all_cif_lines, cif_lines_wyckoffgene
