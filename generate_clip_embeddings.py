from creator_lora.image_encoders.clip import CLIPImageEncoder
from creator_lora.dataset import (
    PickAPicV2Subset,
    UserContextDataset,
)
import os
from creator_lora.utils import create_new_clean_folder

device = "cuda:0"

for i in range(0, 13):
    parquet_filename = os.path.join("downloaded_dataset", f"{i}.pth")
    if i == 0:
        dataset = PickAPicV2Subset(
            parquet_filename=parquet_filename,
        )
    else:
        dataset.append(
            PickAPicV2Subset(
                parquet_filename=parquet_filename,
            )
        )

user_context_dataset = UserContextDataset(pick_a_pic_v2_subset=dataset)

import numpy as np

image_0_uids = np.unique(dataset.pandas_dataframe.image_0_uid.values)
image_1_uids = np.unique(dataset.pandas_dataframe.image_1_uid.values)

pil_images = []
uids = []

for uid in image_0_uids:
    if uid not in uids:
        pil_images.append(dataset.get_image_from_image_0_uid(image_0_uid=uid))
        uids.append(uid)

for uid in image_1_uids:
    if uid not in uids:
        pil_images.append(dataset.get_image_from_image_1_uid(image_1_uid=uid))
        uids.append(uid)

image_encoder = CLIPImageEncoder(name="RN50", device=device)
# ## TODO: encode images based on image uid and not dataset index

image_embeddings_folder = "./clip_embeddings"
clip_embeddings_filenames = [
    os.path.join(image_embeddings_folder, f"{uids[count]}.pth")
    for count in range(len(pil_images))
]

# create_new_clean_folder(image_embeddings_folder)

image_encoder.encode_and_save_batchwise(
    pil_images=pil_images,
    output_filenames=clip_embeddings_filenames,
    batch_size=256,
    skip_if_exists=True
)