import csv
import os
from run_foldx import run_foldx
import shutil

ppb_affinity = []
with open('PPB-Affinity.csv', 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    ppb_affinity = list(reader)

for row in ppb_affinity:
    if row['Source Data Set'] == 'PDBbind v2020':
        pdb = row['PDB']
        pdb_file = f'{pdb.lower()}.ent.pdb'
        output_dir = 'PDBbind-MUT' 
        mutation_file = 'individual_list.txt'
        mutations = row['Mutations']
        mutations = mutations.replace(" ", "")
        print(pdb_file, mutations)
        if os.path.exists(f'PDBbind-MUT/{pdb}_{mutations}.pdb'):
            continue
        if mutations == '':
            shutil.copy2(pdb_file, f'{output_dir}/{pdb}.pdb')
            continue
        mut = mutations.split(',')
        text = ""
        for s in mut:
            s = s[2] + s[0] + s[3:] + ','
            text += s
        text = text[:-1] + ";"
        with open('individual_list.txt', 'w', encoding='utf-8') as file:
            file.write(text)
        foldx = run_foldx(pdb_file, mutation_file, output_dir)
        if not foldx:
            with open('error.txt', 'a', encoding='utf-8') as file:
                file.write(f'{pdb_file}   {mutations}\n')
        if os.path.exists(f'PPB-Affinity/{pdb}_1.pdb'):
            os.rename(f'PDBbind-MUT/{pdb}_1.pdb', f'PDBbind-MUT/{pdb}_{mutations}.pdb')
        


