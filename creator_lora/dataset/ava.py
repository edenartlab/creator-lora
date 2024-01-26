"""
original dataset link:
https://academictorrents.com/details/71631f83b11d3d79d8f84efe0a7e12f0ac001460
"""
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
from ..utils.download import download_torrent
from ..utils.files_and_folders import get_filenames_in_a_folder, extract_7z_multivolume

def download_ava_dataset(
        torrent_url: str = "https://academictorrents.com/details/71631f83b11d3d79d8f84efe0a7e12f0ac001460",
        torrent_filename: str = "ava.torrent",
        download_folder: str = "/data/datasets/aesthetic_visual_analysis/dataset"
):
    assert os.path.exists(download_folder), f"invalid download folder: {download_folder}"
    os.wget(
        torrent_url,
        torrent_filename
    )
    download_torrent(
        torrent_filename,
        save_path=download_folder
    )

def assign_column_names(ava_df):
    """
    Column 1: Index

    Column 2: Image ID 

    Columns 3 - 12: Counts of aesthetics ratings on a scale of 1-10. Column 3 
    has counts of ratings of 1 and column 12 has counts of ratings of 10.

    Columns 13 - 14: Semantic tag IDs. There are 66 IDs ranging from 1 to 66.
    The file tags.txt contains the textual tag corresponding to the numerical
    id. Each image has between 0 and 2 tags. Images with less than 2 tags have
    a "0" in place of the missing tag(s).

    Column 15: Challenge ID. The file challenges.txt contains the name of 
    the challenge corresponding to each ID.
    """
    column_names = [
        "index",
        "image_id"
    ]
    column_names.extend(
        [
            f"num_ratings_{i}" for i in range(1, 11)
        ]
    )

    column_names.extend(
        [
            f"semantic_tag_{i}" for i in range(0, 2)
        ]
    )
    column_names.append("challenge_id")
    ava_df.columns = column_names

    return ava_df

def obtain_mean_rating(ava_df):
    """
    the strategy for now is to take the mean rating for each item
    """
    ratings_columns = [
        f"num_ratings_{i}" for i in range(1, 11)
    ]

    all_num_ratings = [
        ava_df[column].values.reshape(1, -1) for column in ratings_columns
    ]

    # all_ratings.shape: ratings (1-10), num_images
    all_num_ratings = np.concatenate(all_num_ratings, axis = 0)

    rating_values = np.array(
        range(1, 11)
    ).reshape(-1,1)

    mean_ratings = (all_num_ratings*rating_values).sum(0) / all_num_ratings.sum(0)    
    new_data = {
        "image_id": ava_df.image_id.values.tolist(),
        "rating": mean_ratings.tolist()
    }
    return pd.DataFrame(new_data)

def build_ava_dataset(
    download_folder: str,
    extract_image_7z_files: bool = False,
    output_filename = "ava_dataset.csv"
):
    assert os.path.exists(download_folder)
    ava_txt_file = os.path.join(
        download_folder,
        "AVA.txt"
    )
    images_dir = os.path.join(
        download_folder,
        "images"
    )
    assert os.path.exists(ava_txt_file), f"File not found: {ava_txt_file}"
    assert os.path.exists(images_dir), f"File not found: {ava_txt_file}"

    image_7z_files = get_filenames_in_a_folder(folder=images_dir)
    assert len(image_7z_files) == 64

    extracted_images_folder = os.path.join(
        download_folder,
        "extracted_images"
    )

    if extract_image_7z_files:
        print(f"Extracting 7z files:")
        os.chdir(images_dir)
        extract_7z_multivolume(
            name = "images.7z",
            extract_path=extracted_images_folder
        )

    extracted_images_folder = os.path.join(
        extracted_images_folder,
        "images"
    )
    ava_df = pd.read_csv(ava_txt_file, delim_whitespace=True, header=None)
    ava_df = assign_column_names(ava_df = ava_df)
    ava_df = obtain_mean_rating(ava_df)
    
    image_ids = ava_df.image_id.values
    image_filenames = []

    for id in tqdm(image_ids):
        filename = os.path.join(extracted_images_folder, f"{id}.jpg")

        """
        only the lord knows why ~19 images are missing
        """
        try:
            assert os.path.exists(filename), f"Invalid filename: {filename}"
            image_filenames.append(filename)
        except:
            image_filenames.append(None)
        
    ava_df["image_filename"] = image_filenames
    ava_df = ava_df[ava_df['image_filename'].notna()]
    return ava_df.to_csv(output_filename)

class AvaDataset:
    def __init__(self, csv_filename: str, image_transform: callable = None):
        self.df = pd.read_csv(csv_filename)
        self.image_transform=image_transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx: int):

        row = self.df.iloc[idx]

        try:
            image = Image.open(row.image_filename).convert('RGB')
        except OSError:
            """
            criminal hack to deal with that ONE image which is corrupt
            """
            return self.__getitem__(idx+1)
        
        if self.image_transform is not None:
            image = self.image_transform(image)

        return {
            "image": image,
            "label": row.rating
        }