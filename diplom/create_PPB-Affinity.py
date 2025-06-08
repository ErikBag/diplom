import csv 
import os
import shutil


def dir_copy(source_dir, destination_dir, dataset_name):
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir, exist_ok=True)

    for item in os.listdir(source_dir):
        src_path = os.path.join(source_dir, item)
        out_file = dataset_name + '_' + item
        if os.path.isfile(src_path):
            shutil.copy2(src_path, destination_dir + '/' + out_file)

dirs = ['SKEMPI', 'ATLAS', 'PDBbind', 'SAbDab', 'Affinity-Benchmark']
out_dir = 'PPB-Affinity'

for dir in dirs:
    all_dir = f'PDB/{dir}-MUT'
    dir_copy(all_dir, out_dir, dir)