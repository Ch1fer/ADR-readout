from pathlib import Path


def delete_files_in_directory(directory):
    dir_path = Path(directory)

    files = dir_path.glob('*')

    for file in files:
        file.unlink()
