import csv
import os
from run_foldx import run_foldx
import shutil

ppb_affinity = []
with open('PPB-Affinity.csv', 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    ppb_affinity = list(reader)

for row in ppb_affinity:
    if row['Source Data Set'] == 'SAbDab':
        pdb = row['PDB']
        if pdb == '1KBH':
            continue
        pdb_file = f'{pdb.lower()}.pdb'
        output_dir = 'SAbDab-MUT'
        mutation_file = 'individual_list.txt'
        mutations = row['Mutations']
        mutations = mutations.replace(" ", "")
        if os.path.exists(f'SAbDab-MUT/{pdb}_{mutations}.pdb'):
            continue
        if mutations == '':
            shutil.copy2(pdb_file, f'{output_dir}/{pdb}.pdb')
            continue
        mut = mutations.split(',')
        text = ""
        for s in mut:
            s = s[2] + s[0] + s[3:] + ','
            text += s
        s = s[:-1] + ";"
        with open('individual_list.txt', 'w', encoding='utf-8') as file:
            file.write(text)
        foldx = run_foldx(pdb_file, mutation_file, output_dir)
        if not foldx:
            break
        os.rename(f'SAbDab-MUT/{pdb.upper()}_1.pdb', f'SAbDab-MUT/{pdb}_{mutations}.pdb')
        


