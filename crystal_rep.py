from pymatgen.core import Structure
import json
import numpy as np
import sys, pickle
from pymatgen.core import Structure

sys.path.append('/home/energy/mahpe/Published_code/Dis-CSP/dis_csp')
from dis_csp.structure_rep import Crystal_representation

def main():
    
    # Json file with pymatgen structures
    file_json = 'icsd_structure.json' 
    name = 'icsd' # Name of the dataset

    # Load the pymatgen structures from the json file
    with open(file_json) as f:
        data = json.load(f)
    pymatgen_list = [Structure.from_dict(data[i]) for i in range(len(data))]
            

    # Create the crystal representation
    crystal_rep = Crystal_representation(pymatgen_list,verbose=False,
                                        max_wyckoff_sites=9,
                                        max_wyckoff_multiplier=50,
                                        max_element_disorder=6,
                                        max_site_disorder=6, 
                                        remove_P1_spacegroup=True
                                        )
    
    # Save the crystal features, atomic features, and spacegroup to numpy files
    np.save(f'crystal_features_{name}.npy',crystal_rep.total_crystal_features)
    np.save(f'atomic_features_{name}.npy',crystal_rep.total_atomistic_features)
    np.save(f'spacegroup_{name}.npy',crystal_rep.spacegroup_list)

    #Save wyckoff dictionary
    with open(f'wyckoff_dict_{name}.pkl', 'wb') as f:
        pickle.dump(crystal_rep.total_wyckoff_dict, f)

    # Save the symmetrized structures
    with open(f'New_sym_structures_{name}.pkl', 'wb') as f:
        pickle.dump(crystal_rep.total_sym_structures, f)
    
    # Print the total crystal features
    print_value = crystal_rep.total_atomistic_features
    print('---------------------------------------------')
    print('Total crystal features:',crystal_rep.total_crystal_features.shape)
    print('Total number of structures:',print_value.shape[0])
    print('Total atomic representation:',print_value.shape)
    print('One hot:',print_value[:,:,:101].shape, 'Max:',np.max(np.sum(print_value[:,:,:101],axis=2)))
    print('Frac. coordiante:',print_value[:,:,-30:-27].shape, 'Max:',np.max(print_value[:,:,-30:-27]))
    print('Wyckoff multiplier:',print_value[:,:,101:-31].shape, 'Max:',np.max(print_value[:,:,101:-31]))
    print('Wyckoff Letter:',print_value[:,:,-27:].shape, 'Max:',np.max(print_value[:,:,-27:]))
    print('Disordered:',print_value[:,:,-31].shape, 'Max:',np.max(print_value[:,:,-31]))

    total_element_disorder = crystal_rep.total_element_disorder
    total_site_disorder = crystal_rep.total_site_disorder
    print('---------------------------------------------')
    print('Total Element Disorder:',total_element_disorder)
    print('Total Site Disorder:',total_site_disorder)
    print('Total Structures:',len(crystal_rep.total_wyckoff_dict))
    print('Total Lattice Features:',crystal_rep.total_crystal_features.shape)
    print('Total Atomistic Features:',crystal_rep.total_atomistic_features.shape)
    print('---------------------------------------------')
    print('P1 spacegroup removed:',crystal_rep.P1_spacegroup_total_remove)
    print('Max wyckoff sites removed:',crystal_rep.max_wyckoff_sites_total_remove)
    print('Max wyckoff multiplierremoved:',crystal_rep.max_wyckoff_multiplier_total_remove)
    print('Max site disorder removed:',crystal_rep.max_site_disorder_total_remove)
    print('Max element disorder removed:',crystal_rep.max_element_disorder_remove)
    print('Charge disorder removed:',crystal_rep.charge_disorder_total_remove)
    print('---------------------------------------------')

if __name__ == "__main__":
    main()