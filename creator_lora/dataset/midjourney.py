from datasets import load_dataset
from tqdm import tqdm

def prepare_midjourney_dataset(
    images_folder: str, output_json_file: str, max_num_samples: int = None
):
    dataset = load_dataset("wanng/midjourney-v5-202304-clean")["train"]

    """
    steps:
    1. obtain dataset in pandas format
    2. find the number of occurrences of each prompt
    3. filter dataset to contain only prompts which occurred more than once
    4. assert that each prompt is repeated 2 times after filtering
    5. obtain index tuples which map the upscaled image to the non upscaled options image
    6. find out whether the upscaled image matches one of the option images using CLIP similarity (in some cases, they dont)
    7. use CLIP similarity filter the dataset to contain samples which have an options image and a final upscaled image (which was a subset of the options)
    8. use the aspect ratio of the image to split the "options" image into 4 parts i.e 4 images. Label them as: 0=not upscaled and 1=upscaled
    9. save fields: username (str), image_filename (str), label (upscaled=1)
    """

    ## step 1
    dataset.set_format("pandas")

    ## step 2
    ## num_occurrences_for_each_prompt = {prompt1: num_occurences, ...}
    num_occurrences_for_each_prompt = dataset["clean_prompts"].value_counts().to_dict()
    all_prompts = list(num_occurrences_for_each_prompt.keys())

    data = []

    for prompt in tqdm(all_prompts, desc = f"Filtering prompts"):

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

            ## step 5
            if dataset[indices[0]]["upscaled"].values[0] == True:
                upscaled_image_url = dataset[indices[0]]["Attachments"]
                options_image_url = dataset[indices[1]]["Attachments"]
            else:
                upscaled_image_url = dataset[indices[1]]["Attachments"]
                options_image_url = dataset[indices[0]]["Attachments"]


            if dataset[indices[0]]["upscaled"].values[0] == True:
                data.append(
                    {
                        "upscaled": dataset[indices[0]]["Attachments"],
                        "options": dataset[indices[1]]["Attachments"],
                        "prompt": dataset[indices[0]]["clean_prompts"]
                    }
                )
            else:
                data.append(
                    {
                        "upscaled": dataset[indices[1]]["Attachments"],
                        "options": dataset[indices[0]]["Attachments"],
                        "prompt": dataset[indices[0]]["clean_prompts"]
                    }
                )
