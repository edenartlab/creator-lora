from creator_lora.dataset import (
    PickAPicV2Subset,
)
import os

parquet_urls = [
    "https://huggingface.co/datasets/yuvalkirstain/pickapic_v2/resolve/main/data/test-00000-of-00014-387db523fa7e7121.parquet?download=true",
    "https://huggingface.co/datasets/yuvalkirstain/pickapic_v2/resolve/main/data/test-00001-of-00014-b4d27779c32b8591.parquet?download=true",
    "https://huggingface.co/datasets/yuvalkirstain/pickapic_v2/resolve/main/data/test-00002-of-00014-5a7a40ba35ff5c70.parquet?download=true",
    "https://huggingface.co/datasets/yuvalkirstain/pickapic_v2/resolve/main/data/test-00003-of-00014-dddaaa6cb97e4056.parquet?download=true",
    "https://huggingface.co/datasets/yuvalkirstain/pickapic_v2/resolve/main/data/test-00004-of-00014-bda08a373518160d.parquet?download=true",
    "https://huggingface.co/datasets/yuvalkirstain/pickapic_v2/resolve/main/data/test-00005-of-00014-823c2ee536bc1a39.parquet?download=true",
    "https://huggingface.co/datasets/yuvalkirstain/pickapic_v2/resolve/main/data/test-00006-of-00014-a245286301c6ed4a.parquet?download=true",
    "https://huggingface.co/datasets/yuvalkirstain/pickapic_v2/resolve/main/data/test-00007-of-00014-114287665955838d.parquet?download=true",
    "https://huggingface.co/datasets/yuvalkirstain/pickapic_v2/resolve/main/data/test-00008-of-00014-e56083416912a7da.parquet?download=true",
    "https://huggingface.co/datasets/yuvalkirstain/pickapic_v2/resolve/main/data/test-00009-of-00014-d7dfcc392b334773.parquet?download=true",
    "https://huggingface.co/datasets/yuvalkirstain/pickapic_v2/resolve/main/data/test-00010-of-00014-a295e77235e5803c.parquet?download=true",
    "https://huggingface.co/datasets/yuvalkirstain/pickapic_v2/resolve/main/data/test-00011-of-00014-648899a124b34689.parquet?download=true",
    "https://huggingface.co/datasets/yuvalkirstain/pickapic_v2/resolve/main/data/test-00012-of-00014-4261e59adda1963e.parquet?download=true",
    "https://huggingface.co/datasets/yuvalkirstain/pickapic_v2/resolve/main/data/test-00013-of-00014-2dfe413901c5ad39.parquet?download=true"
]

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