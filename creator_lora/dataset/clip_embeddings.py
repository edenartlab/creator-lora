import os
from typing import List, Union
import torch
from ..utils.files_and_folders import get_filenames_in_a_folder

class CLIPEmbeddingsDataset:
    def __init__(self, filenames: List[str], parent_folder = None):
        self.filenames = filenames
        self.parent_folder = parent_folder

    @classmethod
    def from_folder(cls, folder: str):
        assert os.path.exists(folder)
        filenames = get_filenames_in_a_folder(folder=folder)
        return cls(
            filenames=filenames,
            parent_folder=folder
        )

    def __getitem__(self, idx_or_uid: Union[List, str]):

        if isinstance(idx_or_uid, int):
            assert os.path.exists(
                self.filenames[idx_or_uid]
            ), f"Invalid filename: {self.filenames[idx_or_uid]}"
            return torch.load(self.filenames[idx_or_uid])

        else:
            assert self.parent_folder is not None, f"Expected parent_folder to not be None when indexing by uid"
            return torch.load(
                os.path.join(self.parent_folder, f"{idx_or_uid}.pth")
            )


    def __len__(self):
        return len(self.filenames)