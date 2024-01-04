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

"""
10k images = 1.3GB
100k images = 13 GB
500k images = 65 GB
"""

prepare_midjourney_dataset(
    images_folder=images_folder,
    output_json_file=output_json_file,
    max_num_samples=50_000,
    clip_image_encoder_name="ViT-B/32",
    clip_device="cuda:3",
    device="cuda:2",
    clip_similarity_threshold = 0.98,
    resume_from_output_json_file="/data/mayukh/midjourney_dataset/data.json"
)