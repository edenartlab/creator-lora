from creator_lora.dataset import (
    PickAPicV2Subset,
)
import os
from creator_lora.utils import load_json
parquet_urls = load_json("dataset_urls.json")

for i in range(0, len(parquet_urls)):
    parquet_filename = os.path.join("downloaded_dataset", f"{i}.pth")
    if i == 0:
        dataset = PickAPicV2Subset.from_url(
            url=parquet_urls[i].replace("?download=true", ""),
            parquet_filename=parquet_filename,
        )
    else:
        dataset.append(
            PickAPicV2Subset.from_url(
                url=parquet_urls[i].replace("?download=true", ""),
                parquet_filename=parquet_filename,
            )
        )