import os
from collections import Counter
import shutil
from torch_geometric.nn import  radius
from torch_geometric.utils import remove_self_loops
import torch
from tqdm import tqdm

import torch
import numpy as np
from scipy.spatial.distance import cdist

def find_min_distance(vertices):
    """
    Находит наименьшее расстояние между всеми парами вершин в трехмерном пространстве.
    
    Аргументы:
    vertices -- тензор или массив numpy размером Nx3, содержащий координаты вершин
    
    Возвращает:
    min_dist -- наименьшее найденное расстояние
    pair -- индексы пары вершин с минимальным расстоянием (опционально)
    """
    # Преобразуем в numpy массив, если это тензор PyTorch
    if isinstance(vertices, torch.Tensor):
        vertices = vertices.numpy()
    
    # Вычисляем матрицу попарных расстояний
    dist_matrix = cdist(vertices, vertices)
    
    # Заполняем диагональ большими значениями, чтобы исключить расстояние вершины до себя
    np.fill_diagonal(dist_matrix, np.inf)
    
    # Находим минимальное расстояние
    min_dist = np.min(dist_matrix)
    
    # (Опционально) Находим индексы вершин с минимальным расстоянием
    idx = np.argmin(dist_matrix)
    i, j = np.unravel_index(idx, dist_matrix.shape)
    
    return min_dist

def dir_copy(source_dir, destination_dir):
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir, exist_ok=True)  # Создать целевую папку, если её нет

    for item in os.listdir(source_dir):
        src_path = os.path.join(source_dir, item)
        if os.path.isfile(src_path):  # Копируем только файлы
            shutil.copy2(src_path, destination_dir)
    shutil.rmtree(source_dir)

def get_edge_info(edge_index, pos):
    edge_index, _ = remove_self_loops(edge_index)
    j, i = edge_index
    dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()
    return edge_index, dist

def find_coords(file_path):
    coord = []
    if not os.path.exists(file_path):
        return False
    with open(file_path) as f:
        for line in f:
            if '<TRIPOS>ATOM' in line:
                break
        for line in f:
            cont = line.split()
            if '@' in line:
                break
            coor = [float(cont[2]), float(cont[3]), float(cont[4])]
            coord.append(coor)
    return coord


dir = 'data/PPB-Affinity/core-set'
for d in tqdm(os.listdir(dir)):
    print(d)
    all_dir = dir + '/' + d
    lig = []
    poc = []
    for file in os.listdir(all_dir):
        path = all_dir + '/' + file
        if '_ligand' in path:
            lig = find_coords(path)
        if '_pocket' in path:
            poc = find_coords(path)
    complex = np.array(lig + poc)
    lig = np.array(lig) + [2000.0, 0.0, 0.0]
    poc = np.array(poc) + [1000.0, 0.0, 0.0]
    coords = np.concatenate((complex, poc, lig), axis=0)
    coords = torch.tensor(coords)
    row, col = radius(coords, coords, 2.0, max_num_neighbors=1000)
    edge_index_g = torch.stack([row, col], dim=0)
    edge_index_g, dist_g = get_edge_info(edge_index_g, coords)
    if dist_g.min() < 1.0:
            with open('error.txt', 'a', encoding='utf-8') as file:
                file.write(f'{all_dir}\n')
            dir2 = 'data/error-dir/' + dir.split('/')[-1] + '/' + d 
            print(all_dir, dir)
            # dir_copy(all_dir, dir)


# for d in ['data/PPB-Affinity/refined-set/PDBbind_2Z58_[\'B\']_[\'A\']', 
#           'data/PPB-Affinity/refined-set/PDBbind_2IXQ_[\'A\']_[\'B\']',
#           'data/PPB-Affinity/refined-set/PDBbind_4ZRP_[\'A\']_[\'C\']']:
#     file = d.split('/')[-1]
#     dir_copy(d, f'data/{file}')

