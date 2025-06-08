import os
import shutil
import random
import csv

def dir_copy(source_dir, destination_dir):
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir, exist_ok=True)  # Создать целевую папку, если её нет

    for item in os.listdir(source_dir):
        src_path = os.path.join(source_dir, item)
        if os.path.isfile(src_path):  # Копируем только файлы
            shutil.copy2(src_path, destination_dir)

dir = 'data/PPB-Affinity'
out_dir = 'data/PPB-Affinity-mol'
os.makedirs(out_dir, exist_ok=True)
for d in os.listdir(dir):
    if d in ['test', 'train_val']:
        continue
    all_dir = dir + '/' + d
    for file in os.listdir(all_dir):
        path = all_dir + '/' + file
        dir_copy(path, out_dir + '/' + file)

print(f'{out_dir} is created')

ppb_affinity = []
with open('PPB-Affinity.csv', 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    ppb_affinity = list(reader)


dirs = ['SKEMPI', 'ATLAS', 'PDBbind', 'SAbDab', 'Affinity-Benchmark']
dataset_dict = {'PDBbind v2020': 'PDBbind', 
                'SKEMPI v2.0': 'SKEMPI', 
                'Affinity Benchmark v5.5': 'Affinity-Benchmark', 
                'SAbDab': 'SAbDab',
                'ATLAS': 'ATLAS'}

pdbs = []
for row in ppb_affinity:
    pdbs.append(row['PDB'].upper())

pdbs = list(set(pdbs))
sample_size = len(pdbs) // 10
test = random.sample(pdbs, sample_size)

dir = 'data/PPB-Affinity-mol'
out = 'data/PPB-Affinity-new_split_xxx'
os.makedirs(out, exist_ok=True)
for file in os.listdir(dir):
    path = dir + '/' + file
    pdb = file.split('_')[1]
    if pdb in test:
        out_dir = out + '/core-set/' + file
    else:
        out_dir = out + '/refined-set/' + file
    dir_copy(path, out_dir)
    
print(f'{out} is created')