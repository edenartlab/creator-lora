import os
import pandas as pd
from PIL import Image
import io
from typing import List

class PickAPicV2Subset:
    """
    Loads up a small subset of the Pick-a-Pic v2 dataset from a parquet file.

    Example URL: https://huggingface.co/datasets/yuvalkirstain/pickapic_v2/resolve/main/data/test-00000-of-00014-387db523fa7e7121.parquet
    """
    def __init__(self, parquet_filename: str):
        assert os.path.exists(parquet_filename), f"Invalid filename: {parquet_filename}"
        self.pandas_dataframe = pd.read_parquet(parquet_filename, engine='pyarrow')

    def filter_by_user_ids(self, user_ids: List[int]):
        self.pandas_dataframe = self.pandas_dataframe[self.pandas_dataframe.user_id.isin(user_ids)]
    
    @classmethod
    def from_url(cls, url: str, parquet_filename: str, force_redownload = False):
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
            "image_0_uid": row.image_0_uid,
            "image_1_uid": row.image_1_uid,
            "image_0": Image.open(io.BytesIO(row.jpg_0)),
            "image_1": Image.open(io.BytesIO(row.jpg_1)),
            "label_0": row.label_0,
            "label_1": row.label_1,
            "user_id": row.user_id
        }

    def __len__(self):
        return self.pandas_dataframe.shape[0]

    def append(self, dataset: "PickAPicV2Subset"):
        assert isinstance(dataset, PickAPicV2Subset), f"Expected dataset to be an instance of PickAPicV2Subset, but got: {type(dataset)}"
        self.pandas_dataframe = pd.concat(
            [self.pandas_dataframe, dataset.pandas_dataframe]
        )