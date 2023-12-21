import os

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