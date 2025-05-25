from rdkit import Chem
import subprocess
import os
import csv
import json

def fix_and_convert(pdb_file, output_mol2, chains=None):
    """
    Конвертирует PDB файл в MOL2, сохраняя только указанные цепи
    
    Параметры:
        pdb_file (str): Путь к входному PDB файлу
        output_mol2 (str): Путь для сохранения MOL2 файла
        chains (list, optional): Список цепей для сохранения (например ['A', 'B']). 
                                Если None, сохраняются все цепи.
    Возвращает:
        bool: True если конвертация успешна, False в случае ошибки
    """
    try:
        # Загрузка молекулы из PDB
        mol = Chem.MolFromPDBFile(pdb_file, sanitize=False, removeHs=False, proximityBonding=False)
        if mol is None:
            raise ValueError("Не удалось загрузить файл")
        
        # Фильтрация по цепям если указаны
        if chains is not None:
            chains = set(chains)  # Для быстрого поиска
            atoms_to_keep = []
            for atom in mol.GetAtoms():
                res = atom.GetPDBResidueInfo()
                if res and res.GetChainId() in chains:
                    atoms_to_keep.append(atom.GetIdx())
            
            # Создаем подмолекулу только с выбранными цепями
            mol = Chem.RWMol(mol)
            atoms_to_keep = sorted(atoms_to_keep, reverse=True)
            for idx in range(mol.GetNumAtoms()-1, -1, -1):
                if idx not in atoms_to_keep:
                    mol.RemoveAtom(idx)
        
        # Восстановление химической структуры
        Chem.SanitizeMol(mol)
        mol.UpdatePropertyCache()
        
        # Сохранение во временный PDB
        temp_file = "temp_fixed.pdb"
        Chem.MolToPDBFile(mol, temp_file)
        
        # Конвертация через Open Babel
        result = subprocess.run(
            ["obabel", temp_file, "-O", output_mol2], 
            capture_output=True, 
            text=True, 
            check=True
        )
        
        # Удаление временного файла
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
        print(f"Файл успешно сконвертирован: {output_mol2}")
        return True
    
    except Exception as e:
        print(f"\nОшибка при конвертации {pdb_file} в {output_mol2}")
        print(f"Подробности: {str(e)}")
        return False 

def M_Z_fix(filepath,outpath):
    pdb = filepath.split('/')[-1].split('_')[0]
    if not '.pdb' in pdb:
        pdb = pdb + '.pdb'
    with open("M_Z_data.json", "r", encoding="utf-8") as f:
        m_z_chains = json.load(f)
    chain = ''
    if f'PDB/SKEMPI/{pdb}' in m_z_chains:
        chain = m_z_chains[f'PDB/SKEMPI/{pdb}']
    with open(filepath, 'r') as f:
        pdb_data = f.read()
    pdb_data = pdb_data.replace('  MG  MG   ', f' MG    MG {chain}')
    pdb_data = pdb_data.replace('  MN  MN   ', f' MN    MN {chain}')
    pdb_data = pdb_data.replace('  ZN  ZN   ', f' ZN    ZN {chain}')
    with open(outpath, 'w') as f:
        f.write(pdb_data)    
    return outpath

def get_chains(row):
    lig_chains = row['Ligand Chains'].split(',')
    rec_chains = row['Receptor Chains'].split(',')
    if rec_chains == ['']:
        rec_chains = None
    if lig_chains == ['']:
        lig_chains = None
    return rec_chains, lig_chains

def M_Z_chains():
    dirs = ['PDB/SKEMPI/', 'PDB/ATLAS']
    dict = {}
    for dir in dirs:
        for filename in os.listdir(dir):
            if not '.pdb' in filename:
                continue
            filepath = os.path.join(dir, filename)
            with open(filepath, 'r') as f:
                pdb_data = f.read()
            index1 = pdb_data.find('MG    MG')
            index2 = pdb_data.find('ZN    ZN')
            index3 = pdb_data.find('MN    MN')
            index = max(index1, index2, index3)
            if index == -1:
                        continue
            dict[filepath] = pdb_data[index+9]

    with open("M_Z_data.json", "w", encoding="utf-8") as f:
        json.dump(dict, f, ensure_ascii=False, indent=4)


error_file = 'error.txt'

with open(error_file, 'w', encoding='utf-8') as file:
    file.write('')

M_Z_chains()

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
out_dir = 'PPB-Affinity2'

for row in ppb_affinity:
    if row['Subgroup'] == 'Antibody-Antigen':
        out_dir = 'PPB-Affinity2/core-set'
    else:
        out_dir = 'PPB-Affinity2/refined-set'
    dir = dataset_dict[row['Source Data Set']]
    all_dir = f'PDB/{dir}/{dir}-MUT'
    pdb = row['PDB']
    mutations = row['Mutations'].replace(" ", "")
    if mutations == '':
        filename = f'{pdb}.pdb'
    else:    
        filename = f'{pdb}_{mutations}.pdb'
    filepath = os.path.join(all_dir, filename)
    print(filepath)
    if not os.path.exists(filepath):
        print('filepath not exists')
        with open(error_file, 'a', encoding='utf-8') as file:
            file.write(f'{filepath}         filepath not exists\n')
        continue
    rec_chains, lig_chains = get_chains(row)
    outpath = os.path.join(out_dir,dir + '_' + filename[:-4] + '_' + str(rec_chains) + '_' + str(lig_chains))
    if not os.path.exists(outpath):
        os.mkdir(outpath)
    lig_path = os.path.join(outpath, pdb + '_ligand.mol2')
    rec_path = os.path.join(outpath, pdb + '_pocket.mol2')
    inpath = M_Z_fix(filepath, 'fix.pdb')
    if not os.path.exists(lig_path):
        flag = fix_and_convert(inpath, lig_path, lig_chains)
        if not flag:
            with open(error_file, 'a', encoding='utf-8') as file:
                file.write(f'{filepath}         can\'t lig convert\n')
                continue
    if not os.path.exists(rec_path):
        flag = fix_and_convert(inpath, rec_path, rec_chains)
        if not flag:
            with open(error_file, 'a', encoding='utf-8') as file:
                file.write(f'{filepath}         can\'t rec convert\n')


import shutil

dirs = ['core-set', 'refined-set']
error_list = []
for d in dirs:
    dir = 'PPB-Affinity2/' + d
    for mol_dir in os.listdir(dir):
        k = 0
        flag = False
        for file in os.listdir(dir + '/' + mol_dir):
            k+=1
            path = dir +'/' + mol_dir + '/' + file
            if os.path.getsize(path) == 0:
                flag = True
        if k < 2 or flag:
            path = dir + '/' + mol_dir
            shutil.rmtree(path)
            with open(error_file, 'a', encoding='utf-8') as file:
                file.write(f'{path}         error dir created\n')
