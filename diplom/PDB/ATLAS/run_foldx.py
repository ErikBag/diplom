import subprocess
import os

def run_foldx(pdb_file, mutation_file, output_dir):
    # Проверка файлов
    if not all(os.path.exists(f) for f in [pdb_file, mutation_file]):
        raise FileNotFoundError("Проверьте пути к файлам!")

    # Команда FoldX
    command = [
        "foldx_20251231",
        "--command=BuildModel",
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
