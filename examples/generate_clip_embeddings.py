from creator_lora.image_encoders.clip import CLIPImageEncoder
from creator_lora.dataset import (
    PickAPicV2Subset,
    UserContextDataset,
    save_all_unique_images_from_pick_a_pic_v2_subset
)
import os
from creator_lora.utils import create_new_clean_folder
from tqdm import tqdm
import argparse
from creator_lora.utils.json_stuff import save_as_json

device = "cuda:0"

parser = argparse.ArgumentParser(description='Example script with integer arguments')

parser.add_argument('--start-index', type=int, help='start index', default = 0, required = False)
parser.add_argument('--end-index', type=int, help='end index', default = 55, required = False)
args = parser.parse_args()

count = 0
for i in tqdm(range(args.start_index, args.end_index), desc = "Loading data"):
    parquet_filename = os.path.join("downloaded_dataset", f"{i}.pth")
    if count == 0:
        dataset = PickAPicV2Subset(
            parquet_filename=parquet_filename,
        )
    else:
        dataset.append(
            PickAPicV2Subset(
                parquet_filename=parquet_filename,
            )
        )
    count += 1

user_context_dataset = UserContextDataset(pick_a_pic_v2_subset=dataset)

image_paths, uids = save_all_unique_images_from_pick_a_pic_v2_subset(
    pick_a_pic_v2_subset=dataset,
    output_folder="pick_a_pic_images",
    skip_if_exists=False
)

image_encoder = CLIPImageEncoder(name='ViT-B/32', device=device)
# ## TODO: encode images based on image uid and not dataset index

image_embeddings_folder = "./clip_embeddings"
clip_embeddings_filenames = [
    os.path.join(image_embeddings_folder, f"{uids[count]}.pth")
    for count in range(len(image_paths))
]

# create_new_clean_folder(image_embeddings_folder)

image_encoder.encode_and_save_batchwise(
    image_paths=image_paths,
    output_filenames=clip_embeddings_filenames,
    batch_size=256,
    skip_if_exists=False
)

"""
python3 generate_clip_embeddings.py --start-index 0 --end-index 13
CUDA_VISIBLE_DEVICES=1 python3 generate_clip_embeddings.py --start-index 13 --end-index 26
CUDA_VISIBLE_DEVICES=2 python3 generate_clip_embeddings.py --start-index 26 --end-index 39
CUDA_VISIBLE_DEVICES=3 python3 generate_clip_embeddings.py --start-index 39 --end-index 55
"""