import os
from collections import Counter
import shutil

def dir_copy(source_dir, destination_dir):
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir, exist_ok=True)  # Создать целевую папку, если её нет

    for item in os.listdir(source_dir):
        src_path = os.path.join(source_dir, item)
        if os.path.isfile(src_path):  # Копируем только файлы
            shutil.copy2(src_path, destination_dir)

dir = 'data/PPB-Affinity'
out_dir = 'data/PPB-Affinity-mol'
for d in os.listdir(dir):
    all_dir = dir + '/' + d
    for file in os.listdir(all_dir):
        path = all_dir + '/' + file
        dir_copy(path, out_dir + '/' + file)
