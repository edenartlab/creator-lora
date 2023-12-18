import pytest
import os
from creator_lora.dataset import PickAPicV2Subset

parquet_urls = [
    {
        "url": "https://huggingface.co/datasets/yuvalkirstain/pickapic_v2/resolve/main/data/test-00000-of-00014-387db523fa7e7121.parquet",
        "filename": "test.parquet"
    },
    {
        "url": "https://huggingface.co/datasets/yuvalkirstain/pickapic_v2/resolve/main/data/test-00001-of-00014-b4d27779c32b8591.parquet",
        "filename": "test_2.parquet"
    },
]

def run_cleanup():
    for data in parquet_urls:
        os.system(f"rm {data['filename']}")

@pytest.mark.parametrize("url", parquet_urls)
def test_download(url: str):
    dataset = PickAPicV2Subset.from_url(url = url, parquet_filename=parquet_filename)

    assert len(dataset) > 0
    run_cleanup()

def test_append():

    dataset = PickAPicV2Subset.from_url(url = parquet_urls[0]["url"], parquet_filename=parquet_urls[0]["filename"])
    original_len_dataset = len(dataset)
    dataset_2 = PickAPicV2Subset.from_url(url = parquet_urls[1]["url"], parquet_filename=parquet_urls[1]["filename"])

    dataset.append(dataset_2)
    assert len(dataset) == original_len_dataset + len(dataset_2)
    run_cleanup()