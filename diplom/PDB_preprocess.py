import csv
import shutil
import subprocess
import os

def buildModel(pdb_file, pdb_dir, mutation_file, output_dir):
    # Команда FoldX
    command = [
        "foldx_20251231",
        "--command=BuildModel",
        f"--pdb-dir={pdb_dir}",
        f"--pdb={pdb_file}",
        f"--mutant-file={mutation_file}",
        f"--output-dir={output_dir}"
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        print("Успешно! Результаты в:", output_dir)
        return True
    except subprocess.CalledProcessError as e:
        print("Ошибка FoldX:")
        print(e.stderr)
        return False

error_file = 'error_PPB.txt'

with open(error_file, 'w', encoding='utf-8') as file:
    file.write('')

ppb_affinity = []
with open('PPB-Affinity.csv', 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    ppb_affinity = list(reader)

dataset_dict = {'PDBbind v2020': 'PDBbind', 
                'SKEMPI v2.0': 'SKEMPI', 
                'Affinity Benchmark v5.5': 'Affinity-Benchmark', 
                'SAbDab': 'SAbDab',
                'ATLAS': 'ATLAS'}
main_dir = 'PDB'
for row in ppb_affinity:
    dir = dataset_dict[row['Source Data Set']]
    pdb = row['PDB']
    if pdb == '1KBH':
        continue
    if dir == 'PDBbind':
        pdb_file = f'{pdb.lower()}.ent.pdb'
    elif dir == 'SAbDab':
        pdb_file = f'{pdb.lower()}.pdb'
    else:
        pdb_file = f'{pdb}.pdb'
    pdb_dir = main_dir + '/' + dir
    output_dir = main_dir + f'/{dir}-MUT'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    mutation_file = 'individual_list.txt'
    mutations = row['Mutations']
    mutations = mutations.replace(" ", "")
    print(pdb_file, mutations)
    if os.path.exists(f'{output_dir}/{pdb}_{mutations}.pdb'):
        continue
    if mutations == '':
        if os.path.exists(f'{output_dir}/{pdb}.pdb'):
            continue
        shutil.copy2(pdb_dir + '/' + pdb_file, f'{output_dir}/{pdb}.pdb')
        continue
    mut = mutations.split(',')
    text = ""
    for s in mut:
        s = s[2] + s[0] + s[3:] + ','
        text += s
    text = text[:-1] + ";"
    with open('individual_list.txt', 'w', encoding='utf-8') as file:
        file.write(text)
    flag = buildModel(pdb_file,pdb_dir, mutation_file, output_dir)
    if not flag:
        with open(error_file, 'a', encoding='utf-8') as file:
            file.write(f'{dir} {pdb_file}   {mutations}\n')
            continue
    if os.path.exists(f'{output_dir}/{pdb}_1.pdb'):
        os.rename(f'{output_dir}/{pdb}_1.pdb', f'{output_dir}/{pdb}_{mutations}.pdb')
    else:
        with open(error_file, 'a', encoding='utf-8') as file:
            file.write(f'{dir} {pdb_file}   {mutations}\n')