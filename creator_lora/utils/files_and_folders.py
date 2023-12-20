import os

def create_new_clean_folder(folder: str):
    os.system(f"mkdir -p {folder}")
    os.system(f"rm {folder}/*")