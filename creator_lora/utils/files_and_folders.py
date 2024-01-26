import os
from py7zr import SevenZipFile
import multivolumefile

def create_new_clean_folder(folder: str):
    os.system(f"mkdir -p {folder}")
    os.system(f"rm {folder}/*")

def get_filenames_in_a_folder(folder: str):
    """
    returns the list of paths to all the files in a given folder
    """
    
    if folder[-1] == '/':
        folder = folder[:-1]
        
    files =  os.listdir(folder)
    files = [f'{folder}/' + x for x in files]
    return files

def extract_7z_multivolume(name: str, extract_path):
    """
    Extracts content from a multi-volume 7z archive.

    see: https://py7zr.readthedocs.io/en/stable/user_guide.html#extraction-from-multi-volume-archive
    """
    with multivolumefile.open(name, mode='rb') as target_archive:
        with SevenZipFile(target_archive, 'r') as archive:
            archive.extractall(extract_path)