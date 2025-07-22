

from pymatgen.core import  Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core import  Structure
import json
import numpy as np
import pandas as pd
import re
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

def main():
    # Csv file with cif files from ICSD
    file_csv = 'ICSD2024_summary_2024.2_v5.3.0_ascending.csv'
    saved_json = 'icsd_structure.json'  # Json file with pymatgen structures
    saved_npy = 'quary_id_icsd.npy'  # Numpy file with query IDs
    
    # List of elements to drop 
    drop_list = ['Tc','Fm','Md','No','Lr','Rf','Db','Sg','Bh','Hs','Mt','Ds','Rg','Cn','Nh','Fl','Mc','Lv','Ts','Og']


    df_icsd = pd.read_csv(file_csv)
    new_listed_dir = []
    pymatgen_list = []
    pymatgen_list_dict = []
    spacegroup = []
    cifs_icsd = df_icsd['cif']
    quary_ID_icsd = df_icsd['QueryID']
    print('Total number of structures:', len(df_icsd))

    count_not_loaded = 0
    count_not_correct_spacegroup = 0
    count_not_element = 0
    for i in range(len(df_icsd)):
        cif_str = cifs_icsd[i]
        drop = False
        try:
            pymatgen_struc = Structure.from_str(cif_str,fmt='cif');
            sga = SpacegroupAnalyzer(pymatgen_struc,symprec=0.1)
            refined_struc = sga.get_refined_structure()
            sga = SpacegroupAnalyzer(refined_struc,symprec=0.01)
            strc_symmetry = sga.get_symmetrized_structure()
            spacegroup_pymatgen = sga.get_space_group_number()
            
        except:
            print('Loading Error, ', quary_ID_icsd[i])
            count_not_loaded += 1
            continue
        
        match = re.search(r"_space_group_IT_number\s+(\d+)", cif_str)
        spacegroup_icsd = int(match.group(1))

        if spacegroup_icsd != spacegroup_pymatgen:
            print('Not correct spacegroup, ', quary_ID_icsd[i])
            count_not_correct_spacegroup += 1
            continue
        
        for elem in drop_list:
            if elem in pymatgen_struc.formula:
                drop = True 
        if drop:
            count_not_element += 1
            print('Dropped, ', quary_ID_icsd[i])
            continue
        else:
            new_listed_dir.append(quary_ID_icsd[i])
            pymatgen_list_dict.append(pymatgen_struc.as_dict())
            pymatgen_list.append(pymatgen_struc)
            spacegroup.append(spacegroup_pymatgen)
    
    # Save the pymatgen structures to a json file
    with open(saved_json, 'w') as f:
        json.dump(pymatgen_list_dict, f)

    np.save(saved_npy,new_listed_dir) 
    np.save('spacegroup.npy',spacegroup)

if __name__ == "__main__":
    main()