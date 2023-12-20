import os
import pandas as pd
from PIL import Image
import io
from typing import List
from fastparquet import write
import numpy as np
from .clip_embeddings import CLIPEmbeddingsDataset
import torch
from collections import Counter


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

    def get_image_from_image_0_uid(self, image_0_uid: str):
        row = self.pandas_dataframe[
            self.pandas_dataframe.image_0_uid == image_0_uid
        ].iloc[0]

        return Image.open(io.BytesIO(row.jpg_0))

    def get_image_from_image_1_uid(self, image_1_uid: str):
        row = self.pandas_dataframe[
            self.pandas_dataframe.image_1_uid == image_1_uid
        ].iloc[0]

        return Image.open(io.BytesIO(row.jpg_1))

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
        image_uids = []
        pick_a_pic_v2_subset_dataset_indices = []

        for item in single_user_data:
            """
            for now, we take only the first instance of the image if it's repeated
            """

            if item["image_0_uid"] not in image_uids:
                images.append(item["image_0"])
                labels.append(item["label_0"])
                image_uids.append(item["image_0_uid"])

            if item["image_1_uid"] not in image_uids:
                images.append(item["image_1"])
                labels.append(item["label_1"])
                image_uids.append(item["image_1_uid"])

            pick_a_pic_v2_subset_dataset_indices.append(item["dataset_index"])

        num_occurrences_of_image_uids = dict(Counter(image_uids))
        for num_occ in num_occurrences_of_image_uids.values():
            assert (
                num_occ < 2
            ), "Images should not be repeated for a single user in a single context"

        data = {
            "sequence_length": len(images),
            "images": images,
            "labels": labels,
            "user_id": self.user_ids[idx],
            "image_uids": image_uids,
            # "pick_a_pic_v2_subset_dataset_indices": pick_a_pic_v2_subset_dataset_indices,
        }
        return data

    def __len__(self):
        return len(self.user_ids)


class USerContextCLIPEmbeddingsDataset:
    def __init__(
        self,
        user_context_dataset: UserContextDataset,
        clip_embeddings: CLIPEmbeddingsDataset,
    ) -> None:
        self.clip_embeddings = clip_embeddings
        self.user_context_dataset = user_context_dataset
        self.max_sequence_length = self.find_max_sequence_length()

    def find_max_sequence_length(self):
        sequence_lengths = [len(x["image_uids"]) for x in self.user_context_dataset]
        return max(sequence_lengths)

    def __getitem__(self, idx) -> dict:
        data = self.user_context_dataset[idx]
        data["labels"] = torch.tensor(data["labels"])
        image_embeddings = []
        for uid in data["image_uids"]:
            image_embeddings.append(self.clip_embeddings[uid])

        ## shape: 1, seq, emb
        data["image_embeddings"] = torch.cat(image_embeddings, dim=0)

        assert data["image_embeddings"].shape[0] == len(data["labels"])

        del data["image_uids"]
        del data["images"]
        # del data["labels"]

        return data

    def __len__(self):
        return len(self.user_context_dataset)

    @staticmethod
    def collate_fn_with_padding(batch):
        """
        to be used as a collate_fn for a dataloader on top of USerContextCLIPEmbeddingsDataset
        """
        max_sequence_length = max([b['sequence_length'] for b in batch])
        """
        apply padding
        """
        for data in batch:
            if data["image_embeddings"].shape[0] < max_sequence_length:
                num_pad_tokens = (
                    max_sequence_length - data["image_embeddings"].shape[0]
                )
                data["image_embeddings"] = torch.cat(
                    [
                        data["image_embeddings"],
                        torch.zeros(num_pad_tokens, data["image_embeddings"].shape[1])
                    ],
                    dim=0,
                )
                ## set label to -1 for pad token indices
                data["labels"] = torch.cat(
                    [
                        data["labels"],
                        torch.zeros(num_pad_tokens) - 1
                    ],
                    dim=0,
                )
            data["image_embeddings"] = data["image_embeddings"].unsqueeze(0)
            data["labels"] =  data["labels"].unsqueeze(0)

        return {
            "image_embeddings": torch.cat([data["image_embeddings"] for data in batch]),
            "labels": torch.cat([data["labels"] for data in batch]),
            "sequence_lengths": [b['sequence_length'] for b in batch],
            "user_ids": [b['user_id'] for b in batch]
        }