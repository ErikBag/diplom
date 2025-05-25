import os
import numpy as np
from tqdm import tqdm
from openbabel import pybel
from scipy.spatial import distance
import csv

import torch
from torch_geometric.nn import radius
from torch_geometric.utils import remove_self_loops

from utils.featurizer import Featurizer


def find_interacting_atoms(protein_coords, ligand_coords, cutoff):
    dist_matrix = distance.cdist(protein_coords, ligand_coords)

    protein_indices, ligand_indices = np.where(dist_matrix <= cutoff)

    close_ligand_indices = list(np.unique(ligand_indices))
    close_protein_indices = list(np.unique(protein_indices))

    return close_protein_indices, close_ligand_indices

def pocket_atom_num_from_mol2(name, path):
    n = 0
    name2 = name.split('_')[1]
    with open('%s/%s/%s_pocket.mol2' % (path, name, name2)) as f:
        for line in f:
            if '<TRIPOS>ATOM' in line:
                break
        for line in f:
            cont = line.split()
            if '@' in line or cont[7] == 'HOH':
                break
            n += int(cont[5][0] != 'H')
    return n

def construct_graphs(data_dir, save_dir, data_name, save_name, label_dict, cutoff, exclude_data_name=None):
    """
    For each ligand-protein complex, a graph G is constructed by concatenating 3 subgraphs:
    1. Complex subgraph: pocket & ligand
    2. Pocket subgraph: pocket + shift 100 angstroms along the x-axis
    3. Ligand subgraph: ligand + shift 200 angstroms along the x-axis

    The choice of 100/200 angstroms is to use a distance >> the scale of any complex in our dataset.
    By doing so, all 3 subgraphs are far away from each other in 3D space, and have no interactions.
    Then the message passings can be applied to the subgraphs in parallel by simply loading G.
    """
    pybel.ob.obErrorLog.StopLogging()
    print("Preprocessing", data_name)

    # Get list of directories to be excluded if needed
    if exclude_data_name != None and exclude_data_name != 'PDBbind':
        exclude_dir = os.path.join(data_dir, exclude_data_name)
        exclude_name_list = []
        for dir_name in os.listdir(exclude_dir):
            if dir_name not in ['index', 'readme']:
                exclude_name_list.append(dir_name.split('_')[1])
    
    if exclude_data_name == 'PDBbind':
        exclude_dir = 'data/PDBbind/refined-set'
        not_exclude_dir = 'data/PDBbind/core-set'
        exclude_name_list = []
        for dir_name in os.listdir(exclude_dir):
            if dir_name not in ['index', 'readme']:
                exclude_name_list.append(dir_name.upper())
        not_exclude_name_list = []
        for dir_name in os.listdir(not_exclude_dir):
            if dir_name not in ['index', 'readme']:
                not_exclude_name_list.append(dir_name.upper())
        exclude_name_list = list(set(exclude_name_list) - set(not_exclude_name_list))
        
    
    # Get list of directories for constructing graphs
    data_dir_full = os.path.join(data_dir, data_name)
    
    name_list = []
    for dir_name in os.listdir(data_dir_full):
        if dir_name not in ['index', 'readme']:
            if exclude_data_name != None:
                if dir_name not in exclude_name_list and dir_name.split('_')[1] not in exclude_name_list:
                    name_list.append(dir_name)
            else:
                name_list.append(dir_name)

    save_dir_full = os.path.join(save_dir, save_name, "raw")
    
    if not os.path.exists(save_dir_full):
        os.makedirs(save_dir_full)

    for file_name in [save_name + '_node_labels.txt', save_name + '_graph_indicator.txt', 
                save_name + '_node_attributes.txt', save_name + '_graph_labels.txt']:
        if os.path.isfile(os.path.join(save_dir_full, file_name)):
            os.remove(os.path.join(save_dir_full, file_name))

    for i in tqdm(range(len(name_list))):
        name = name_list[i]
        name2 = name.split('_')[1]
        pdb_label = label_dict[name]
        pdb_label = np.array(pdb_label).reshape(-1, 1)

        featurizer = Featurizer(save_molecule_codes=False)

        charge_idx = featurizer.FEATURE_NAMES.index('partialcharge')

        ligand = next(pybel.readfile('mol2', os.path.join(data_dir_full, name, name2 + '_ligand.mol2')))
        ligand_coords, ligand_features = featurizer.get_features(ligand, molcode=1)

        pocket = next(pybel.readfile('mol2', os.path.join(data_dir_full, name, name2 + '_pocket.mol2')))
        pocket_coords, pocket_features = featurizer.get_features(pocket, molcode=-1)

        node_num = pocket_atom_num_from_mol2(name, data_dir_full)
        pocket_coords = pocket_coords[:node_num]
        pocket_features = pocket_features[:node_num]

        assert (ligand_features[:, charge_idx] != 0).any()
        assert (pocket_features[:, charge_idx] != 0).any()
        assert (ligand_features[:, :9].sum(1) != 0).all()
        assert ligand_features.shape[0] == ligand_coords.shape[0]
        assert pocket_features.shape[0] == pocket_coords.shape[0]
        
        pocket_interact, ligand_interact = find_interacting_atoms(pocket_coords, ligand_coords, cutoff)
        cut = cutoff
        while len(pocket_interact) + len(ligand_interact) < 30:
            cut *= 1.5
            pocket_interact, ligand_interact = find_interacting_atoms(pocket_coords, ligand_coords, cut)

        if len(pocket_interact) + len(ligand_interact) > 2500:
            print(name, name2)
            print(cutoff)
            print(len(pocket_interact), len(ligand_interact))
            exit()
        pocket_atoms = set([])
        pocket_atoms = pocket_atoms.union(set(pocket_interact))        
        pocket_atoms = np.array(list(pocket_atoms))

        pocket_coords = pocket_coords[pocket_atoms]
        pocket_features = pocket_features[pocket_atoms]

        ligand_atoms = set([])
        ligand_atoms = ligand_atoms.union(set(ligand_interact))
        ligand_atoms = np.array(list(ligand_atoms))
        
        ligand_coords = ligand_coords[ligand_atoms]
        ligand_features = ligand_features[ligand_atoms]

        ligand_pos = np.array(ligand_coords)
        pocket_pos = np.array(pocket_coords)

        # Concat three subgraphs:
        complex_pos = np.concatenate((pocket_pos, ligand_pos), axis=0)
        complex_features = np.concatenate((pocket_features, ligand_features), axis=0)
        
        x_shift = np.mean(complex_pos[:, 0])
        complex_pos -= [x_shift, 0.0, 0.0]
        pocket_pos -= [x_shift, 0.0, 0.0]
        ligand_pos -= [x_shift, 0.0, 0.0]

        pocket_pos += [1000.0, 0.0, 0.0]    # shift 100 angstroms along the x-axis
        ligand_pos += [2000.0, 0.0, 0.0]    # shift 200 angstroms along the x-axis

        final_pos = np.concatenate((complex_pos, pocket_pos, ligand_pos), axis=0)
        final_features = np.concatenate((complex_features, pocket_features, ligand_features), axis=0)

        # Generate files for loading graphs
        indicator = np.ones((final_features.shape[0], 1)) * (i + 1)

        with open(os.path.join(save_dir_full, save_name + '_graph_indicator.txt'),'ab') as f:
            np.savetxt(f, indicator, fmt='%i', delimiter=', ')
        f.close()
    
        with open(os.path.join(save_dir_full, save_name + '_node_labels.txt'),'ab') as f:
            np.savetxt(f, final_features, fmt='%.4f', delimiter=', ')
        f.close()
 
        with open(os.path.join(save_dir_full, save_name + '_node_attributes.txt'),'ab') as f:
            np.savetxt(f, final_pos, fmt='%.3f', delimiter=', ')
        f.close()
        
        with open(os.path.join(save_dir_full, save_name + '_graph_labels.txt'),'ab') as f:
            np.savetxt(f, pdb_label, fmt='%.2f', delimiter=', ')
        f.close()


def get_chains(row):
    lig_chains = row['Ligand Chains'].split(',')
    rec_chains = row['Receptor Chains'].split(',')
    if rec_chains == ['']:
        rec_chains = None
    if lig_chains == ['']:
        lig_chains = None
    return rec_chains, lig_chains

def get_path(row):
    dataset_dict = {'PDBbind v2020': 'PDBbind', 
                'SKEMPI v2.0': 'SKEMPI', 
                'Affinity Benchmark v5.5': 'Affinity-Benchmark', 
                'SAbDab': 'SAbDab',
                'ATLAS': 'ATLAS'}

    if row['Subgroup'] == 'Antibody-Antigen':
        dir = 'PPB-Affinity/core-set'
    else:
        dir = 'PPB-Affinity/refined-set'

    dataset = dataset_dict[row['Source Data Set']]
    pdb = row['PDB']
    mutations = row['Mutations'].replace(" ", "")
    if mutations == '':
        filename = f'{pdb}.pdb'
    else:    
        filename = f'{pdb}_{mutations}.pdb'
    rec_chains, lig_chains = get_chains(row)
    path = os.path.join(dir,dataset + '_' + filename[:-4] + '_' + str(rec_chains) + '_' + str(lig_chains))
    return path


def main():
    data_dir = os.path.join(".", "data", "PPB-Affinity3")    
    # Loaded lines have format:
    # PDB code, resolution, release year, -logKd/Ki, Kd/Ki, reference, ligand name
    # The base-10 logarithm, -log kd/pk
    
    ppb_affinity = []
    with open('PPB-Affinity.csv', 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        ppb_affinity = list(reader)
    
    label_dict = {}
    for row in ppb_affinity:
        path = get_path(row)
        if os.path.exists('data/' + path):
            path = path.split('/')[-1]
            label_dict[path] = - (8.314/4184)*(273.15 + 25.0) * np.log(float(row['KD(M)']))
    
    cutoff = 6.0

    # Use core-set as testing set
    # Use refined-set (excluding core-set) as training+validation set
    construct_graphs(data_dir, data_dir, "core-set", "test", label_dict, cutoff, "PDBbind")
    construct_graphs(data_dir, data_dir, "refined-set", "train_val", label_dict, cutoff, "core-set")


if __name__ == "__main__":
    main()