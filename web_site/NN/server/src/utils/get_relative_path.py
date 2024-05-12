from pathlib import Path


def get_file_path_in_module(file_name: str, current_file: Path) -> Path:
    current_folder_path = current_file.resolve().parent

    file_path_in_module = current_folder_path / file_name

    return file_path_in_module
