import os
from tqdm import tqdm
from pinscrape import pinscrape
from typing import List
from ..utils.json_stuff import save_as_json
from ..utils.files_and_folders import create_folder_if_it_doesnt_exist, get_filenames_in_a_folder


def build_pinterest_dataset(
    keys: List[str],
    download_folder: str,
    output_json_file: str,
    max_num_images_per_key = 10_000,
    num_threads=10,
):
    assert os.path.exists(download_folder), f"Invalid download folder: {download_folder}"

    image_urls = []
    image_filenames = []
    image_keys = []

    for key in tqdm(keys):
        try:
            output_folder = os.path.join(
                download_folder, key.replace(' ', '')
            )
            create_folder_if_it_doesnt_exist(output_folder)
            details = pinscrape.scraper.scrape(
                key=key, 
                output_folder=output_folder, 
                proxies={}, 
                threads=num_threads, 
                max_images = max_num_images_per_key
            )

            image_urls.extend(details["extracted_urls"])
            image_filenames.extend(get_filenames_in_a_folder(output_folder))
            image_keys.extend([key for i in range(len(image_filenames))])

            for filename in image_filenames:
                assert os.path.exists(filename), f"Invalid image filename: {filename}"

            if details["isDownloaded"]:
                pass
            else:
                print(f"Nothing to download for key: {key}")
                
        except KeyboardInterrupt:
            print(f"Detected keyboard interrupt, stopping download...")
            break

        print(f"Saved {len(image_filenames)} images so far")

    data = {
        "key": image_keys,
        "filename": image_filenames,
        "url": image_urls 
    }
    print(f"Total number of images: {len(data['filename'])}")
    save_as_json(
        data,
        output_json_file
    )