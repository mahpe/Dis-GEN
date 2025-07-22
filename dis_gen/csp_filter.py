import numpy as np
import smact
from smact import element_dictionary
import itertools 
import smact
from smact.screening import pauling_test

def structure_validity(crystal, cutoff=0.5):
    """
    Check if the crystal structure is valid based on distance matrix and volume.
    A valid crystal structure should have a minimum distance between atoms greater than the cutoff
    and a positive volume.
    Args:
        crystal (Crystal): The crystal structure to check.
        cutoff (float): The minimum distance between atoms to consider the structure valid.
    Returns:
        bool: True if the structure is valid, False otherwise.
    """
    dist_mat = crystal.distance_matrix
    # Pad diagonal with a large number
    dist_mat = dist_mat + np.diag(
        np.ones(dist_mat.shape[0]) * (cutoff + 10.))
    if dist_mat.min() < cutoff or crystal.volume < 0.1:
        return False
    else:
        return True

def oxidation_state_validity(struc,two_oxidation_state=True,use_pauling_test=True,verbose=False,pauling_threshold=0.2):
    """
    Check if the structure has valid oxidation states based on the smact package.
    Args:
        struc (Structure): The structure to check.
        two_oxidation_state (bool): If True, allows elements to have two different oxidation states.
        use_pauling_test (bool): If True, uses the Pauling electronegativity test.
        verbose (bool): If True, prints additional information.
        pauling_threshold (float): Threshold for the Pauling electronegativity test.
    Returns:
        bool: True if the structure has valid oxidation states, False otherwise.
    """
    # DOES NOT WORK: For charge disorder, meaning an element cannot have multiple oxidation states in the same structure
    #                For when an anion and cation have same electronegativity 
    #                For disordered structures with partial occupancies rounded to zero 
    # Load the composition
    composition = struc.composition
    
    if verbose:
        print(composition)

    # Find the unique elements
    elem_symbols = tuple(composition.as_dict().keys())
    if verbose:
        print(elem_symbols)
    # Find the count of each element
    count = tuple(composition.as_dict().values())
    count = [int(np.round(c)) for c in count]
    if any([c == 0 for c in count]): # some disordered structures have partial occupancies rounded to zero
        if verbose:
            print('count',count)
            print('Disordered structure, scaling composition by 100')
        composition = composition*100 # multiply by 100 to get rid of the rounding error
        count = tuple(composition.as_dict().values())
        count = [int(np.round(c/1)) for c in count] 

    if verbose:
        print(count)
    # See if the count can be reduced by a common factor
    if len(count) == 1:
        # If there is only one element, we can just return True
        if verbose:
            print('Only one element, returning True')
        return True
    gcd = smact._gcd_recursive(*count)
    if verbose:
        print('gcd:',gcd)
    count = [int(c / gcd) for c in count]

    # Get the space of elements and define the smact elements
    space = element_dictionary(elem_symbols)
    smact_elems = [e[1] for e in space.items()]
    # Get the electronegativities and all possible oxidation states for each element
    electronegs = [e.pauling_eneg for e in smact_elems]
    ox_combos = [e.oxidation_states_icsd24 for e in smact_elems] # usees icsd24 oxidation states

    # Need to make it possible for the elements to have two different oxidation states
    if two_oxidation_state:
        count += count 
        elem_symbols += elem_symbols
        ox_combos += ox_combos
        electronegs += electronegs

    if verbose:
        print(count)
        print(elem_symbols)
        print('electronegs:',electronegs)
        print('ox_combos:',ox_combos)
    # Set the threshold of the maximum number of atoms
    threshold = np.max(count)
    compositions = []
    use_pauling_test = True
    # Loop over all possible oxidation states
    for ox_states in itertools.product(*ox_combos):
       
        stoichs = [(c,) for c in count]
        
        # Test for charge balance
        cn_e, cn_r = smact.neutral_ratios(ox_states, stoichs=stoichs, threshold=threshold)
        
        # Electronegativity test
        if cn_e:
            if verbose:
                print('ox_states:',ox_states)
                print('stoichs:',stoichs)
                print('cn_e:',cn_e,'cn_r:',cn_r)
            if use_pauling_test:
                try:
                    electroneg_OK = pauling_test(ox_states, electronegs,threshold=pauling_threshold)
                except TypeError:
                    # if no electronegativity data, assume it is okay
                    electroneg_OK = True
            else:
                electroneg_OK = True
            if electroneg_OK:
                for ratio in cn_r:
                    compositions.append((elem_symbols, ox_states, ratio))
    compositions = [(i[0], i[2]) for i in compositions]
    compositions = list(set(compositions))
    return len(compositions) > 0

def Symmetry_matching_accuracy(wyckoff_letter,wyckoff_multiplier,spacegroup):
    """
    Compare the multiplicity and letter of the wyckoff sites from a VAE model with the true letter and multiplicity given for a specific spacegroup.
    Args:
        
        wyckoff_letter (list): List of letters from the VAE model.
        wyckoff_multiplier (list): List of multiplicities from the VAE model.
        spacegroup (pyxtal.symmetry.Group): The spacegroup object containing the true multiplicities and letters.
        verbose (bool): If True, prints additional information.
    Returns:
        bool: True if the multiplicities and letters match, False otherwise.
    """

    # Assert that the length of the wyckoff_letter and wyckoff_multiplier are equal
    if len(wyckoff_letter) != len(wyckoff_multiplier):
        return False
    
    # Possible wyckoff sites Symmetry accucary 
    symmetry_accuracy = False

    for vae_letter, vae_mult in zip(wyckoff_letter, wyckoff_multiplier):
        try:
            mult_true = spacegroup.get_wp_by_letter(vae_letter).multiplicity
        except IndexError:
            symmetry_accuracy = False
            break

        if mult_true != vae_mult:      
            symmetry_accuracy = False
            break 
        
        else: 
            symmetry_accuracy = True
    
    return symmetry_accuracy


