import os
import pandas as pd
from PIL import Image
import io
from typing import List
from fastparquet import write
import numpy as np
from .clip_embeddings import CLIPEmbeddingsDataset
import torch

class PickAPicV2Subset:
    """
    Loads up a small subset of the Pick-a-Pic v2 dataset from a parquet file.

    Example URL: https://huggingface.co/datasets/yuvalkirstain/pickapic_v2/resolve/main/data/test-00000-of-00014-387db523fa7e7121.parquet
    """

    def __init__(self, parquet_filename: str):
        assert os.path.exists(parquet_filename), f"Invalid filename: {parquet_filename}"
        self.pandas_dataframe = pd.read_parquet(parquet_filename, engine="pyarrow")

    def filter_by_user_ids(self, user_ids: List[int]):
        self.pandas_dataframe = self.pandas_dataframe[
            self.pandas_dataframe.user_id.isin(user_ids)
        ]

    def filter_by_image_0_uids(self, image_0_uids: List[str]):
        self.pandas_dataframe = self.pandas_dataframe[
            self.pandas_dataframe.image_0_uid.isin(image_0_uids)
        ]

    def filter_by_image_1_uids(self, image_1_uids: List[str]):
        self.pandas_dataframe = self.pandas_dataframe[
            self.pandas_dataframe.image_1_uid.isin(image_1_uids)
        ]

    @classmethod
    def from_url(cls, url: str, parquet_filename: str, force_redownload=False):
        if force_redownload:
            os.system(f"wget -O {parquet_filename} {url}")
        else:
            if os.path.exists(parquet_filename) is not True:
                os.system(f"wget -O {parquet_filename} {url}")
            else:
                pass

        return cls(parquet_filename=parquet_filename)

    def __getitem__(self, idx):
        row = self.pandas_dataframe.iloc[idx]

        return {
            "dataset_index": idx,
            "image_0_uid": row.image_0_uid,
            "image_1_uid": row.image_1_uid,
            "image_0": Image.open(io.BytesIO(row.jpg_0)),
            "image_1": Image.open(io.BytesIO(row.jpg_1)),
            "label_0": row.label_0,
            "label_1": row.label_1,
            "user_id": row.user_id,
        }

    def get_all_data_for_a_single_user(self, user_id: int) -> List[dict]:
        # Get the indices where the values are True
        bool_mask = (self.pandas_dataframe.user_id == user_id).values
        indices = np.where(bool_mask)[0]
        assert (
            len(indices) > 0
        ), f"Found zero items for user_id: {user_id}, Are you sure you're providing the correct user_id?"
        return [self.__getitem__(idx) for idx in indices]

    def __len__(self):
        return self.pandas_dataframe.shape[0]

    def append(self, dataset: "PickAPicV2Subset"):
        assert isinstance(
            dataset, PickAPicV2Subset
        ), f"Expected dataset to be an instance of PickAPicV2Subset, but got: {type(dataset)}"
        self.pandas_dataframe = pd.concat(
            [self.pandas_dataframe, dataset.pandas_dataframe]
        )

    def save(self, parquet_filename: str, append=False):
        # https://stackoverflow.com/a/73775084
        write(parquet_filename, self.pandas_dataframe, append=append)


class UserContextDataset:
    def __init__(self, pick_a_pic_v2_subset: PickAPicV2Subset):
        self.pick_a_pic_v2_subset = pick_a_pic_v2_subset

        ## store IDs of all users
        self.user_ids = np.unique(
            self.pick_a_pic_v2_subset.pandas_dataframe.user_id.values
        )

    def __getitem__(self, idx):
        """
        return all images and labels for a single user as a list
        """
        single_user_data = self.pick_a_pic_v2_subset.get_all_data_for_a_single_user(
            user_id=self.user_ids[idx]
        )

        images = []
        labels = []
        pick_a_pic_v2_subset_dataset_indices = []

        for item in single_user_data:
            images.append(item["image_0"])
            labels.append(item["label_0"])
            images.append(item["image_1"])
            labels.append(item["label_1"])

            pick_a_pic_v2_subset_dataset_indices.append(item["dataset_index"])
        data = {
            "images": images,
            "labels": labels,
            "user_id": self.user_ids[idx],
            "pick_a_pic_v2_subset_dataset_indices": pick_a_pic_v2_subset_dataset_indices,
        }
        return data

    def __len__(self):
        return len(self.user_ids)

class USerContextCLIPEmbeddingsDataset:
    def __init__(
        self,
        user_context_dataset: UserContextDataset,
        clip_embeddings_image_0: CLIPEmbeddingsDataset,
        clip_embeddings_image_1: CLIPEmbeddingsDataset,
    ) -> None:
        assert len(user_context_dataset.pick_a_pic_v2_subset) == len(
            clip_embeddings_image_0
        )
        assert len(user_context_dataset.pick_a_pic_v2_subset) == len(
            clip_embeddings_image_1
        )

        self.clip_embeddings_image_0 = clip_embeddings_image_0
        self.clip_embeddings_image_1 = clip_embeddings_image_1
        self.user_context_dataset = user_context_dataset

    def __getitem__(self, idx) -> dict:
        data = self.user_context_dataset[idx]

        image_embeddings = []
        for idx in data["pick_a_pic_v2_subset_dataset_indices"]:
            image_embeddings.append(self.clip_embeddings_image_0[idx])
            image_embeddings.append(self.clip_embeddings_image_1[idx])

        data["image_embeddings"] = torch.cat(image_embeddings, dim=0)

        assert data["image_embeddings"].shape[0] == len(data["labels"])
        return data

    def __len__(self):
        return len(self.user_context_dataset)