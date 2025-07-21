from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core import Structure
from typing import List, Tuple
import re, json
import numpy as np
from keras.utils import to_categorical

class Crystal_representation:
    def __init__(
            self,
            pymatgen_list: List[Structure] ,
            verbose:bool=False,
            remove_P1_spacegroup = True,
            max_wyckoff_sites = None,
            max_site_disorder = None,
            max_element_disorder = None,
            max_wyckoff_multiplier = None
    ):
        """
        Class to represent the crystal structure in a format that can be used for machine learning
        Args:
            pymatgen_list: List of pymatgen structure objects
            verbose: Print the features
            remove_P1_spacegroup: Remove the P1 spacegroup
            max_wyckoff_sites: Maximum number of wyckoff sites
            max_site_disorder: Maximum Wyckoff site with disorder
            max_element_disorder: Maximum number of element in a Wyckoff site disorder
            max_wyckoff_multiplier: Maximum wyckoff multiplier
        """
        
        # Initialize the input parameters
        self.pymatgen_list = pymatgen_list
        self.verbose = verbose
        self.max_wyckoff_sites = max_wyckoff_sites
        self.remove_P1_spacegroup = remove_P1_spacegroup
        self.max_site_disorder = max_site_disorder
        self.max_element_disorder = max_element_disorder
        self.max_wyckoff_multiplier = max_wyckoff_multiplier
        
        # Initialize the total site disorder and wyckoff sites
        self.total_site_disorder = 0
        self.total_element_disorder = 0
        self.total_wyckoff_sites = 0
        self.total_wyckoff_multiplier = 0
        

        # Initialize the filter size
        self.P1_spacegroup_total_remove = 0
        self.max_wyckoff_sites_total_remove = 0
        self.max_site_disorder_total_remove = 0
        self.max_element_disorder_remove = 0
        self.max_wyckoff_multiplier_total_remove = 0
        self.charge_disorder_total_remove = 0

        # Initialize the spacegroup list
        self.spacegroup_list = []

        # Initialize max_site_disorder_list
        self.max_site_disorder_list = []

        # Initialize the max_element_disorder_list
        self.max_element_disorder_list = []

        # Initialize the total wyckoff multiplier list
        self.total_wyckoff_multiplier_list = []

        # Initialize the total sym_structures
        self.total_sym_structures = []

        # Initialize the elemental features
        self.get_element_features()

        # Get the crystal and wyckoff features
        self.WyckCrust_representation()

    
    def WyckCrust_representation(self)-> None:

        # Loop through all the structures
        self.total_crystal_features = []
        self.total_wyckoff_dict = []
        count = 0 
        for pymatgen_struc in self.pymatgen_list:
            #print('Structure:',pymatgen_struc.composition.reduced_formula,count)
            count += 1
            # Symmetrize the structure
            strc_symmetry, spacegroup = self.symmetrize_symmetrized_structure(pymatgen_struc)
            # Get the crystal features
            self.total_crystal_features.append(self.get_crystal_features(pymatgen_struc,spacegroup))
            # Get the wyckoff dictionary
            self.total_wyckoff_dict.append(self.get_wyckoff_dict(strc_symmetry))
            # Append the symmetrized structure
            self.total_sym_structures.append(strc_symmetry)
        
        # Get the atomistic features by looping through the wyckoff dictionary
        self.total_atomistic_features = []
        self.total_mask_tensor = []
        self.drop_index = []

        # Limit the maximum number of wyckoff sites 
        if self.max_wyckoff_sites:
            if self.max_wyckoff_sites < self.total_wyckoff_sites:
                self.total_wyckoff_sites = self.max_wyckoff_sites

        # Limit the maximum site disorder 
        if self.max_site_disorder:
            if self.max_site_disorder < self.total_site_disorder:
                self.total_site_disorder = self.max_site_disorder
        
        # Limit the maximum element disorder
        if self.max_element_disorder:
            if self.max_element_disorder < self.total_element_disorder:
                self.total_element_disorder = self.max_element_disorder

        # Limit the maximum wyckoff multiplier if specified
        if self.max_wyckoff_multiplier:
            if self.max_wyckoff_multiplier < self.total_wyckoff_multiplier:
                self.total_wyckoff_multiplier = self.max_wyckoff_multiplier

        # Loop through all the wyckoff dictionaries and get the atomistic features
        for idx, wyckoff_dict in enumerate(self.total_wyckoff_dict):

            # Filter the structures based on the spacegroup
            if self.remove_P1_spacegroup:
                if self.spacegroup_list[idx] == 1:
                    self.drop_index.append(idx)
                    self.P1_spacegroup_total_remove += 1
                    #print('P1_spacegroup:',idx)
                    continue
            
            # Filter the structures based on the maximum number of wyckoff sites
            if len(wyckoff_dict) > self.total_wyckoff_sites:
                self.drop_index.append(idx)
                self.max_wyckoff_sites_total_remove += 1
                #print('Max_wyckoff_site:',idx)
                continue
            
            # Filter the structures based on the maximum site disorder
            max_site_disorder = self.max_site_disorder_list[idx]
            if max_site_disorder > self.total_site_disorder:
                self.drop_index.append(idx)
                self.max_site_disorder_total_remove += 1
                #print('Max_site_disorder:',idx)
                continue

            # Filter the structures based on the maximum element disorder
            max_element_disorder = self.max_element_disorder_list[idx]
            if max_element_disorder > self.total_element_disorder:
                self.drop_index.append(idx)
                self.max_element_disorder_remove += 1
                #print('Max_site_disorder:',idx)
                continue
            
            # Filter the structures based on the maximum wyckoff multiplier
            max_wyckoff_multiplier = self.total_wyckoff_multiplier_list[idx]
            if max_wyckoff_multiplier > self.total_wyckoff_multiplier:
                self.drop_index.append(idx)
                self.max_wyckoff_multiplier_total_remove += 1
                #print('Max_wyckoff_multiplier:',idx)
                continue

            # Filter charge disorder states
            charge_disorder = [y['charge_disorder'] for x,y in wyckoff_dict.items()]
            if any(charge_disorder):
                self.drop_index.append(idx)
                self.charge_disorder_total_remove += 1
                #print('Charge Disorder:',idx)
                continue 
            
            atomic_features = self.get_atomistic_features(wyckoff_dict)
            self.total_atomistic_features.append(atomic_features)

            continue

        # Drop the structures that do not meet the criterias
        self.total_crystal_features = np.array([y for i,y in enumerate(self.total_crystal_features) if i not in self.drop_index])
        self.total_wyckoff_dict = [y for i,y in enumerate(self.total_wyckoff_dict) if i not in self.drop_index]

        # Drop the structures that do not meet the criterias
        self.spacegroup_list = [y for i,y in enumerate(self.spacegroup_list) if i not in self.drop_index]
        self.total_sym_structures = [y for i,y in enumerate(self.total_sym_structures) if i not in self.drop_index]

        self.total_atomistic_features = np.array(self.total_atomistic_features)

    def symmetrize_symmetrized_structure(self,pymatgen_struc:Structure,) -> Tuple[Structure,int]:
        """
        Symmetrize the structure and get the spacegroup number
        Args:
            pymatgen_struc: pymatgen structure object
        Returns:
            strc_symmetry: Symmetrized structure
            spacegroup: Spacegroup number
        
        """

        # Symmetry analysis 
        sga = SpacegroupAnalyzer(pymatgen_struc, symprec=0.1)
        strc_conv = sga.get_refined_structure()
        sga = SpacegroupAnalyzer(strc_conv, symprec=0.01)
        strc_symmetry = sga.get_symmetrized_structure()
        spacegroup = sga.get_space_group_number()
        self.spacegroup_list.append(spacegroup)

        return strc_symmetry, spacegroup

    def get_element_features(self)-> None:
        """
        Get the elemental features from the data folder
        """
        
        # Load the cgcnn embedding
        elem_embedding_file = 'data/atom_init.json'
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        self.elem_embedding = {int(key): value for key, value
                            in elem_embedding.items()}
        
        feat_cgcnn = []
        for key, value in self.elem_embedding.items():
            feat_cgcnn.append(value)
        self.feat_cgcnn = np.array(feat_cgcnn)
                
        # Elemental categorical features with 1 element extra for nothingness
        self.E_v = to_categorical(np.arange(0, self.feat_cgcnn.shape[0]+1, 1))
        

    def get_crystal_features(self,pymatgen_struc:Structure,spacegroup:int)-> np.array:
        """
        Function to get the crystal features
        Args:
            pymatgen_struc: pymatgen structure object
            spacegroup: Spacegroup number
        Returns:
            crystal_features: Crystal features of lattice constants and spacegroup
        """

        # Get the lattice information
        lattice = pymatgen_struc.lattice
        abc = np.array(lattice.abc) # either array or 3x3 matrix
        ang = np.array(lattice.angles)

        assert len(abc) == 3, 'Lattice constants should be 3'
        assert len(ang) == 3, 'Lattice angles should be 3'
        lattice_constants = np.concatenate((abc, ang), axis=0)

        # Ont-hot crystal system featurizer
        sg_cat = np.zeros((230))
        sg_cat[spacegroup-1] = 1
        sg_cat = sg_cat
        sg_cat_list = sg_cat

        # Append lattice constants and spacegroup
        crystal_features = np.concatenate((lattice_constants, sg_cat_list), axis=0)
        if self.verbose:
            print('Lattice Constants:', lattice_constants.shape)
            print('Spacegroup:',sg_cat_list.shape)
            print('Crystal Features:',crystal_features.shape)
            print('---------------------------------')

        return crystal_features
    

    def get_wyckoff_dict(self,strc_symmetry:Structure)-> dict:
        """
        Get the wyckoff dictionary
        Args:
            strc_symmetry: Symmetrized structure
        Returns:
            wyckoff_dict: Dictionary of the wyckoff positions, species and fractional coordinates
        """
        wyckoff_dict = {}
        total_wyckoff_sites = 0
        max_site_disorder = 0
        max_element_disorder = 0
        max_wyckoff_multiplier = 0
        # Loop through all the equivalent sites
        for idx, sites in enumerate(strc_symmetry.equivalent_sites):
            site = sites[0]

            # Get the wyckoff position, species and fractional coordinates
            wyckoff_position = strc_symmetry.wyckoff_symbols[idx]
            species_dict = site.species.get_el_amt_dict()
            frac_coords = site.frac_coords
            try:
                Z_species = np.array([int(element.number) for element in site.species.elements ])
                wrong_site = False
            except:
                Z_species = None
                print('Error:',site.species.elements)
                print('Site:',site.species)
                wrong_site = True # if the site is wrong, then it is a charge disorder

            # Check if some of the Z_species are the same
            if wrong_site:
                charge_disorder = True
            else:
                if len(Z_species) != len(np.unique(Z_species)):
                    charge_disorder = True
                else:
                    charge_disorder = False
            
            # Update the total element disorder
            if len(species_dict) > self.total_element_disorder:
                self.total_element_disorder = len(species_dict)
            
            # Update the total wyckoff multiplier
            if int(wyckoff_position[:-1]) > self.total_wyckoff_multiplier:
                self.total_wyckoff_multiplier = int(wyckoff_position[:-1])

            # Update the local max element disorder
            if len(species_dict) > max_element_disorder:
                max_element_disorder = len(species_dict)

            # Update the local max site disorder
            if len(sites) > 1:
                max_site_disorder += 1
            
            # Update the local max wyckoff multiplier
            if int(wyckoff_position[:-1]) > max_wyckoff_multiplier:
                max_wyckoff_multiplier = int(wyckoff_position[:-1])
            

            # Add the information to the dictionary
            wyckoff_dict[idx] = {'wyckoff':wyckoff_position,'species': species_dict,
                                  'frac_coords': frac_coords, 'Z_species': Z_species,
                                  'charge_disorder': charge_disorder
                                }

            total_wyckoff_sites += 1
        
        # Update the total number of wyckoff sites
        if total_wyckoff_sites > self.total_wyckoff_sites:
            self.total_wyckoff_sites = total_wyckoff_sites
        
        # Update the total site disorder
        if max_site_disorder > self.total_site_disorder:
            self.total_site_disorder = max_site_disorder

        # Append the local max element disorder
        self.max_element_disorder_list.append(max_element_disorder)

        # Append the local max site disorder
        self.max_site_disorder_list.append(max_site_disorder)

        # Append the local max wyckoff multiplier to the list
        self.total_wyckoff_multiplier_list.append(max_wyckoff_multiplier)

        return wyckoff_dict

    
    def get_atomistic_features(self,wyckoff_dict_struc:dict)-> np.array:
        """
        Function to get the atomistic features
        Args:
            wyckoff_dict: Dictionary of the wyckoff positions, species and fractional coordinates for the whole structure
        Returns:
            stucture_feature: Atomistic features of the categorical, CGCNN, fractional coordinates, wyckoff letter and wyckoff multiplier
        """

        # Create feature matrix
        stucture_feature = []
        # Loop through all the wyckoff sites
        for idx,wyckoff_dict in wyckoff_dict_struc.items():
            #print('Wyckoff:',wyckoff_dict['wyckoff'])   
            # Extract the information from the wyckoff dictionary
            wyckoff = wyckoff_dict['wyckoff'] # Wyckoff position
            species = wyckoff_dict['species'] # Element species and occupancy
            occupancy_species = np.array([species[atom_element] for atom_element in species.keys()]) # Occupancy of the species
            Z_species = wyckoff_dict['Z_species'] # Atomic number of the species
            N_species = len(species) # Total number of species

            # Check if the sum of occupancy is 1
            if np.sum(occupancy_species) != 1:
                # add vacancy to the end of the species
                occ_X = 1 - np.sum(occupancy_species)
                species['X'] = occ_X
                Z_species = np.append(Z_species,0)
                occupancy_species = np.append(occupancy_species,occ_X)
                N_species += 1

            #assert np.sum(occupancy_species) == 1, 'Sum of occupancy should be 1'

            #print(Z_species,occupancy_species,species)

            # Catagorical layer
            onehot = self.E_v[:, Z_species - 1]*occupancy_species # index with 0, where X is the last element 
            if N_species > 1: # if there is more than 1 species than sum the onehot so its 1 dimensional
                onehot = np.sum(onehot,axis=1)
            else:
                onehot = onehot.flatten()
            #assert np.sum(onehot) == 1, 'Sum of onehot should be 1'            

            # Fractional coordinates layer
            frac_coords = np.array(wyckoff_dict['frac_coords'])

            # Wyckoff multiplier layer, where the multiplicity is multiplied by the occupancy of the species
            wyckoff_multiplier = np.zeros((self.total_wyckoff_multiplier+1)) # maximum wyckoff multiplier + 1 for zero padding
            wyckoff_multiplier[int(wyckoff[:-1])] = 1 # index 1 is equal to 1, index 2 is equal to 2, etc. index 0 is equal to 0
            #wyckoff_multiplier = np.array([int(wyckoff[:-1])])

            # Wyckoff letter layer
            wyckoff_letter = np.zeros((27)) # 26 letters in the alphabet + 1 for zero padding
            site_num = ord(re.sub('[^a-zA-Z]+', '',wyckoff )) - 96 # index 1 is equal to a, index 2 is equal to b, etc. index 0 is equal to 0
            wyckoff_letter[site_num] = 1
           
            if len(Z_species) >1 :
                disordered = np.array([1])
            else:
                disordered = np.array([0])


            # Concatenate all the features
            atomistic_features = np.concatenate((onehot, wyckoff_multiplier,disordered,frac_coords, wyckoff_letter ), axis=0)
            # Append the features to the structure feature list
            stucture_feature.append(atomistic_features)

            if self.verbose:
                print('Wyckoff:', wyckoff)
                print('Species:', species)
                print('Occupancy:', occupancy_species)
                print('Z:', Z_species)
                print('N:', N_species)
                print('Disordered:', disordered)
                print('Onehot:', onehot.shape, onehot)
                print('Frac_coords:', frac_coords.shape, frac_coords.T)
                print('Wyckoff_multiplier:',wyckoff_multiplier.shape, wyckoff_multiplier.T)
                print('Wyckoff Letter:', wyckoff_letter.shape, wyckoff_letter.T)
                print('Atomistic Features:', atomistic_features.shape)
                print('---------------------------------')

        # Transpose the structure feature 
        stucture_feature = np.array(stucture_feature).T
        # Reshape the structure feature to the maximum number of wyckoff sites and pad with zeros
        reshaped_structure_feature = np.zeros((atomistic_features.shape[0],self.total_wyckoff_sites))
        
        # Add a ones in the zero padding to indicate that there is no atom
        reshaped_structure_feature[100,:] = 1 # Onehot at index -1
        reshaped_structure_feature[101,:] = 1 # Wyckoff multiplier at index 0
        reshaped_structure_feature[-27,:] = 1 # Wyckoff letter at index 0
        
        # Add the structure feature to the reshaped structure feature    
        reshaped_structure_feature[:,:stucture_feature.shape[1]] = stucture_feature

        # Swap the two axis to get the correct shape for PyTorch (wyckoff_sites, features) for Tensorflow the channel dim are in the end, while for PyTorch is in the start 
        reshaped_structure_feature = np.swapaxes(reshaped_structure_feature,0,1)

        if self.verbose:
            print('Structure Feature:', reshaped_structure_feature.shape)
            print('---------------------------------')


        return reshaped_structure_feature