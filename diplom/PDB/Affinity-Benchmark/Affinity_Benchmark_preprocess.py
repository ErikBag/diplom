import csv
import os
from run_foldx import run_foldx
import shutil

ppb_affinity = []
with open('PPB-Affinity.csv', 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    ppb_affinity = list(reader)

for row in ppb_affinity:
    if row['Source Data Set'] == 'Affinity Benchmark v5.5':
        pdb = row['PDB']
        pdb_file = f'{pdb}.pdb'
        output_dir = 'Affinity_Benchmark-MUT'
        mutation_file = 'individual_list.txt'
        mutations = row['Mutations']
        mutations = mutations.replace(" ", "")
        print(pdb_file, mutations)
        if os.path.exists(f'Affinity_Benchmark-MUT/{pdb}_{mutations}.pdb'):
            continue
        if mutations == '':
            shutil.copy(pdb_file, output_dir)
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
            os.rename(f'Affinity_Benchmark-MUT/{pdb}_1.pdb', f'Affinity_Benchmark-MUT/{pdb}_{mutations}.pdb')
        


