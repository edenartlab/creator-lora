import os
import pandas as pd
from tqdm import tqdm
import wget
import time
from PIL import Image
from ..utils.json_stuff import load_json, save_as_json

def convert_oid_to_string_in_df(df):
    df.user = [x["$oid"] for x in df.user.values]
    df.creation = [x["$oid"] for x in df.creation.values]
    return df

def parse_user_data(
    creations_data_path: str,
    reactions_data_path: str
):
    ## load existing data and convert them to pandas dataframe objects
    deletion_data = load_json(creations_data_path)
    reaction_data = load_json(reactions_data_path)
    deletion_df = pd.DataFrame(deletion_data)
    reaction_df = pd.DataFrame(reaction_data)

    ## convert values like {"$oid": x} to x for user and creation columns
    deletion_df = convert_oid_to_string_in_df(deletion_df)
    reaction_df = convert_oid_to_string_in_df(reaction_df)

    ## keep only items IDs which were deleted
    deletion_df = deletion_df[deletion_df["deleted"] == True]

    ## find user IDs which are present in both the deletion and reaction df
    deletion_unique_users = deletion_df.user.unique()
    reaction_unique_users = reaction_df.user.unique()
    common_users = [x for x in deletion_unique_users if x in reaction_unique_users]

    ## keep only image reactions and remove video reactions
    reaction_df = reaction_df[reaction_df.creationUri.str.endswith("jpg")]

    ## filter both dfs by common users
    reaction_df = reaction_df[reaction_df["user"].isin(common_users)]
    deletion_df = deletion_df[deletion_df["user"].isin(common_users)]

    final_dataset = {
        "user": [],
        "filename":[],
        "data_type": [],
        "activity": [],
        "url": []
    }

    for row_idx in range(len(reaction_df)):
        final_dataset["user"].append(reaction_df.user.values[row_idx])
        final_dataset["filename"].append(None)
        final_dataset["url"].append(reaction_df.creationUri.values[row_idx])
        final_dataset["data_type"].append("video" if reaction_df.creationUri.values[row_idx].endswith("mp4") else "image")
        final_dataset["activity"].append(reaction_df.reaction.values[row_idx])

    for row_idx in range(len(deletion_df)):
        final_dataset["user"].append(deletion_df.user.values[row_idx])
        final_dataset["filename"].append(None)
        final_dataset["url"].append(deletion_df.creationUri.values[row_idx])
        final_dataset["data_type"].append("video" if deletion_df.creationUri.values[row_idx].endswith("mp4") else "image")
        final_dataset["activity"].append("delete")

    return final_dataset

def download_with_retries(url, filename, max_retries=10):
    for attempt in range(1, max_retries + 1):
        try:
            # Download the image using wget
            wget.download(url, filename)
            return  # Exit the loop if download is successful
        except Exception as e:
            print(f"\nError downloading on attempt {attempt}: {e}")
            time.sleep(1)  # Wait for 1 second before retrying

    print(f"\nFailed to download after {max_retries} attempts")

def build_eden_dataset(
    creations_data_path: str,
    reactions_data_path: str,
    images_folder: str,  ## all images would be downloaded here
    output_filename: str ## would contain the final_dataset json
):
    parsed_dataset = parse_user_data(
        creations_data_path=creations_data_path,
        reactions_data_path=reactions_data_path
    )

    assert os.path.exists(images_folder)

    for dataset_idx in tqdm(
        range(len(parsed_dataset["user"])), 
        desc = "downloading images"
    ):
        url = parsed_dataset['url'][dataset_idx]

        image_filename = os.path.join(
            images_folder, f"{dataset_idx}.jpg"
        )

        if os.path.exists(image_filename) is not True:
            download_with_retries(
                url = url,
                filename = image_filename
            )
        parsed_dataset["filename"][dataset_idx] = image_filename

    save_as_json(
        parsed_dataset,
        output_filename
    )

class EdenDataset:
    def __init__(self, filename: str):
        self.data = load_json(filename)
        self.labels_map = {
            "praise": 1,
            "delete": 0
        }

    def __len__(self):
        return len(self.data["user"])

    def get_label_from_activity(self, activity: str):
        assert activity in list(self.labels_map.keys())
        return self.labels_map[activity]

    def __getitem__(self, idx: int):
        image_filename = self.data["filename"][idx]
        assert os.path.exists(image_filename), f"Invalid image path: {image_filename}"

        return {
            "user": self.data["user"][idx] ,
            "image": Image.open(image_filename),
            "label": self.get_label_from_activity(self.data["activity"][idx])
        }