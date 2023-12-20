from creator_lora.image_encoders.clip import CLIPImageEncoder
from creator_lora.dataset import (
    PickAPicV2Subset,
    UserContextDataset,
)
import os
from creator_lora.utils import create_new_clean_folder
from tqdm import tqdm

device = "cuda:0"

for i in tqdm(range(0, 35), desc = "Loading data"):
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

def save_all_unique_images_from_pick_a_pic_v2_subset(pick_a_pic_v2_subset: PickAPicV2Subset, output_folder: str, skip_if_exists: bool):
    image_0_uids = np.unique(pick_a_pic_v2_subset.pandas_dataframe.image_0_uid.values)
    image_1_uids = np.unique(pick_a_pic_v2_subset.pandas_dataframe.image_1_uid.values)

    uids = []
    filenames = []
    for uid in tqdm(image_0_uids, desc = "saving pil images [image_0]"):
        
        if uid not in uids:
            filename = os.path.join(output_folder, f"{uid}.jpg")
            filenames.append(filename)
            if os.path.exists(filename) and skip_if_exists:
                continue
            else:
                dataset.get_image_from_image_0_uid(image_0_uid=uid).save(
                    filename
                )
            uids.append(uid)

    for uid in tqdm(image_1_uids, desc = "saving pil images [image_1]"):
        
        if uid not in uids:
            filename = os.path.join(output_folder, f"{uid}.jpg")
            filenames.append(filename)
            if os.path.exists(filename) and skip_if_exists:
                continue
            else:
                dataset.get_image_from_image_1_uid(image_1_uid=uid).save(
                    filename
                )
            uids.append(uid)

    return filenames, uids

image_paths, uids = save_all_unique_images_from_pick_a_pic_v2_subset(
    pick_a_pic_v2_subset=dataset,
    output_folder="pick_a_pic_images",
    skip_if_exists=True
)

image_encoder = CLIPImageEncoder(name='ViT-B/32', device=device)
# ## TODO: encode images based on image uid and not dataset index

image_embeddings_folder = "./clip_embeddings"
clip_embeddings_filenames = [
    os.path.join(image_embeddings_folder, f"{uids[count]}.pth")
    for count in range(len(image_paths))
]

create_new_clean_folder(image_embeddings_folder)

image_encoder.encode_and_save_batchwise(
    image_paths=image_paths,
    output_filenames=clip_embeddings_filenames,
    batch_size=256,
    skip_if_exists=True
)