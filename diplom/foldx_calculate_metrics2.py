import subprocess
import os
import csv
import numpy as np
from math import sqrt
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr
import math


def rmse(y,f):
    rmse = sqrt(((y - f)**2).mean(axis=0))
    return rmse

def mae(y,f):
    mae = (np.abs(y-f)).mean()
    return mae

def pearson(y,f):
    rp = np.corrcoef(y, f)[0,1]
    return rp

def spearman_corr(y, f):
    sc = spearmanr(y, f)[0]
    return sc

def optimize(pdb_dir, pdb_file, out_dir):
    command = [
        "foldx_20251231",
        "--command=Optimize",
        f"--pdb-dir={pdb_dir}",
        f"--pdb={pdb_file}",
        f"--output-dir={out_dir}"
    ]
    print(command)
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print("Ошибка Optimize:")
        print(e.stderr)
        return False
    
def analyseComplex(pdb_dir, pdb_file, lig_chains, rec_chains, out_dir, out_file):
    command = [
        "foldx_20251231",
        "--command=AnalyseComplex",
        f"--pdb-dir={pdb_dir}",
        f"--pdb={pdb_file}",
        f"--output-dir={out_dir}",
        f"--output-file={out_file}",
        f"--analyseComplexChains={lig_chains},{rec_chains}"
    ]
    print(command)
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print("Ошибка AnalyseComplex:")
        print(e.stderr)
        return False
    
def get_chains(row):
    lig_chains = row['Ligand Chains'].replace(" ", "").split(',')
    rec_chains = row['Receptor Chains'].replace(" ", "").split(',')
    lig_chains_str = ""
    for i in lig_chains:
        lig_chains_str += i
    rec_chains_str = ""
    for i in rec_chains:
        rec_chains_str += i
    return rec_chains_str, lig_chains_str


error_file = 'error4.txt'

with open(error_file, 'w', encoding='utf-8') as file:
    file.write('')

dataset_dict = {'PDBbind v2020': 'PDBbind', 
                'SKEMPI v2.0': 'SKEMPI', 
                'Affinity Benchmark v5.5': 'Affinity-Benchmark', 
                'SAbDab': 'SAbDab',
                'ATLAS': 'ATLAS'}

ppb_affinity = []
with open('PPB-Affinity.csv', 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    ppb_affinity = list(reader)

interactionEnergy = []
experimentEnergy = []

pdb_dir = 'PPB-Affinity'
optimize_dir= 'test_optimize2'
out_dir = 'test2'

pdbs = []
dir = 'PAMNet/data/PPB-Affinity-new_split/core-set'
for file in os.listdir(dir):
    pdb = file.split('_')[1]
    pdbs.append(pdb)

for row in ppb_affinity:
    if row['PDB'] not in pdbs:
        continue
    mutations = row['Mutations'].replace(" ", "")
    pdb = row['PDB']
    dataset = dataset_dict[row['Source Data Set']]  
    temp_str = row['Temperature(K)']
    if temp_str != '':
        temp = float(row['Temperature(K)'][:3])
    else:
        temp = 298
    if mutations == '':
        filename = f'{dataset}_{pdb}.pdb'
    else:    
        filename = f'{dataset}_{pdb}_{mutations}.pdb'
    filepath = os.path.join(pdb_dir, filename)
    if not os.path.exists(filepath):
        continue
    rec_chains, lig_chains = get_chains(row)
    pdb_optimize = 'Optimized_' + filename
    if not os.path.exists(optimize_dir + '/' + pdb_optimize):
        flag = optimize(pdb_dir, filename, optimize_dir)
        if not flag:
            with open(error_file, 'a', encoding='utf-8') as file:
                file.write(f'{filename}         can\'t optimize\n')
            continue
    if not os.path.exists(optimize_dir + '/' + pdb_optimize):
        with open(error_file, 'a', encoding='utf-8') as file:
            file.write(f'{filename}         can\'t optimize\n')
        continue
    out_file = filename[:-4] + '_' + rec_chains + '_' + lig_chains
    pdb_analyse = 'Summary_' + out_file + '_AC.fxout'
    if not os.path.exists(out_dir + '/' + pdb_analyse):
        flag = analyseComplex(optimize_dir, pdb_optimize, lig_chains, rec_chains, out_dir, out_file)
        if not flag:
            with open(error_file, 'a', encoding='utf-8') as file:
                file.write(f'{filename}         can\'t analyseComplex\n')
            continue
    if not os.path.exists(out_dir + '/' + pdb_analyse):
        with open(error_file, 'a', encoding='utf-8') as file:
            file.write(f'{filename}         can\'t analyseComplex\n')
        continue
    with open(out_dir + '/' + pdb_analyse, "r", encoding="utf-8") as file:
        lines = file.readlines()
        line_10 = lines[9] 
        cont = line_10.split()
        energy1 = float(cont[5])
    energy2 = (8.314/4184)* temp * np.log(float(row['KD(M)']))
    if math.isnan(energy1) or math.isnan(energy2):
        continue
    interactionEnergy.append(energy1)
    experimentEnergy.append(energy2)

print(interactionEnergy)
print(experimentEnergy)
pred = np.array(interactionEnergy)
y = np.array(experimentEnergy)
print(f'RMSE: {rmse(y, pred)}, MAE: {mae(y, pred)}, SRCC: {spearman_corr(y, pred)}, PCC: {pearson(y, pred)}')


