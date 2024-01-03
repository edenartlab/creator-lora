from creator_lora.dataset.midjourney import prepare_midjourney_dataset
import os
root_folder = "/data/mayukh/midjourney_dataset"

images_folder = os.path.join(
    root_folder,
    "images"
)

output_json_file = os.path.join(
    root_folder,
    "data.json"
)

prepare_midjourney_dataset(
    images_folder=images_folder,
    output_json_file=output_json_file,
    max_num_samples=50_000
)