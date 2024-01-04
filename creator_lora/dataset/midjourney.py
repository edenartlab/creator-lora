import os
from datasets import load_dataset
from tqdm import tqdm
from typing import Callable
from ..utils.image import load_pil_image, split_pil_image_into_quadrants
from ..image_encoders.clip import CLIPImageEncoder
from ..utils.json_stuff import save_as_json, load_json
import torch.nn.functional as F

def prepare_midjourney_dataset(
    images_folder: str, output_json_file: str, max_num_samples: int = 100,
    clip_image_encoder_name =  "ViT-B/32", clip_device = "cuda:3", device = "cuda:2",
    clip_similarity_threshold: float = 0.98
):
    dataset = load_dataset("wanng/midjourney-v5-202304-clean")["train"]
    clip_image_encoder = CLIPImageEncoder(
        name = clip_image_encoder_name,
        device = clip_device
    )

    """
    steps:
    1. obtain dataset in pandas format
    2. find the number of occurrences of each prompt
    3. filter dataset to contain only prompts which occurred more than once
    4. assert that each prompt is repeated 2 times after filtering
    5. obtain index tuples which map the upscaled image to the non upscaled options image
    6. find out whether the upscaled image matches one of the option images using CLIP similarity (in some cases, they dont)
    7. use CLIP similarity filter the dataset to contain samples which have an options image and a final upscaled image (which was a subset of the options)
    8. save fields: username (str), image_filename (str), label (upscaled=1), prompt (str)
    """

    ## step 1
    dataset.set_format("pandas")

    ## used to name images
    image_index = 0

    ## step 2
    ## num_occurrences_for_each_prompt = {prompt1: num_occurences, ...}
    num_occurrences_for_each_prompt = dataset["clean_prompts"].value_counts().to_dict()
    all_prompts = list(num_occurrences_for_each_prompt.keys())

    data = []

    pbar = tqdm(len(all_prompts))
    for prompt in all_prompts:
        if image_index >= max_num_samples:
            print(f"Already saved max_num_samples: {max_num_samples} images.")
            break
        pbar.update(1)
        ## step 3
        if num_occurrences_for_each_prompt[prompt] == 2:
            indices = (dataset["clean_prompts"] == prompt).values.nonzero()[0].tolist()

            ## step 4
            assert (
                len(indices) == 2
            ), f"Expected 2 indices but got: {len(indices)} indices"

            ## making sure that one of the images is "upscaled" and the other one is the options image
            assert (
                dataset[indices[0]]["upscaled"].values[0] == True
                and dataset[indices[1]]["upscaled"].values[0] == False
            ) or (
                dataset[indices[0]]["upscaled"].values[0] == False
                and dataset[indices[1]]["upscaled"].values[0] == True
            )

            ## make sure that its the same user
            try:
                assert dataset[indices[0]]["user_name"].values[0] == dataset[indices[1]]["user_name"].values[0]
            except:
                print(f'Skipping because of mismatch in username ({dataset[indices[0]]["user_name"].values[0]} != {dataset[indices[1]]["user_name"].values[0]}) even though the prompt is the same: {prompt}')
                continue

            ## step 5
            if dataset[indices[0]]["upscaled"].values[0] == True:
                upscaled_image_url = dataset[indices[0]]["Attachments"].values[0]
                options_image_url = dataset[indices[1]]["Attachments"].values[0]
            else:
                upscaled_image_url = dataset[indices[1]]["Attachments"].values[0]
                options_image_url = dataset[indices[0]]["Attachments"].values[0]

            try:
                upscaled_pil_image = load_pil_image(upscaled_image_url)
                options_pil_image = load_pil_image(options_image_url)
            except:
                print(f"Skipping because: could not retrieve image(s) from url")
                continue

            options_pil_images_split = split_pil_image_into_quadrants(
                image=options_pil_image
            )
            embeddings = clip_image_encoder.encode(
                pil_images=[upscaled_pil_image, *options_pil_images_split],
                batch_size = 4,
                progress = False
            )
            cosine_similarities = F.cosine_similarity(
                x1 = embeddings[0, :].unsqueeze(0).to(device),
                x2 = embeddings[1:, :].to(device),
            ).tolist()

            ## step 6 and step 7
            if max(cosine_similarities) >= clip_similarity_threshold:
                index_of_matching_image_in_options =  cosine_similarities.index(max(cosine_similarities))
                labels = [0 for i in range(len(cosine_similarities))]
                labels[index_of_matching_image_in_options] = 1

                for image, label in zip(options_pil_images_split, labels):
                    filename = os.path.join(
                        images_folder,
                        f"index_{image_index}_label_{label}.jpg"
                    )
                    image.save(filename)
                    ## step 8
                    data.append(
                        {
                            "image_filename": filename,
                            "label": label,
                            "prompt": dataset[indices[0]]["clean_prompts"].values[0],
                            "username": dataset[indices[0]]["user_name"].values[0]
                        }
                    )
                    image_index += 1
                    pbar.set_description(f"Saved: {image_index} images")
            else:
                pass

    save_as_json(
        data,
        filename = output_json_file
    )

class MidJourneyDataset:
    def __init__(self, images_folder: str, output_json_file: str, image_transform: Callable = None):
        assert os.path.exists(images_folder)
        assert os.path.exists(output_json_file)

        self.data = load_json(
            filename = output_json_file
        )
        self.image_transform=image_transform

    def __getitem__(self, idx: int):
        item = self.data[idx]
        image = load_pil_image(item["image_filename"])
        if self.image_transform is not None:
            image = self.image_transform(image)
        else:
            pass
        return {
            "image": image,
            "label": item["label"],
            "prompt": item["prompt"],
            "username": item["username"]
        }

    def __len__(self):
        return len(self.data)