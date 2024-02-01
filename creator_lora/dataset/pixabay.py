import requests
import os
from tqdm import tqdm
import time
from typing import List
from ..utils.json_stuff import save_as_json
from ..utils.image import load_pil_image
from ..utils.files_and_folders import (
    create_folder_if_it_doesnt_exist,
    get_filenames_in_a_folder,
)


def pixabay_api_request(
    api_key,
    query,
    lang="en",
    image_type="all",
    orientation="all",
    category=None,
    min_width=0,
    min_height=0,
    colors=None,
    editors_choice=False,
    safesearch=False,
    order="popular",
    page=1,
    per_page=20,
    callback=None,
    pretty=False,
):
    # Pixabay API endpoint
    api_url = "https://pixabay.com/api/"

    # Construct parameters for the request
    params = {
        "key": api_key,
        "q": query,
        "lang": lang,
        "image_type": image_type,
        "orientation": orientation,
        "category": category,
        "min_width": min_width,
        "min_height": min_height,
        "colors": colors,
        "editors_choice": "true" if editors_choice else "false",
        "safesearch": "true" if safesearch else "false",
        "order": order,
        "page": page,
        "per_page": per_page,
        "callback": callback,
        "pretty": "true" if pretty else "false",
    }

    # Make the API request
    response = requests.get(api_url, params=params)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse and return the JSON response
        return response.json()
    else:
        # Print the error message if the request was unsuccessful
        print(f"Error: {response.status_code}")
        return None


def build_pixabay_image_dataset(
    keys: List[str],
    download_folder: str,
    api_key: str,
    output_json_file: str,
    min_height: int = 512,
    min_width: int = 512,
    order: str = "popular",
    per_page: int = 20,
    num_pages=25,
):
    """
    keys: list of search terms like flowers, cats, dogs, etc
    """

    image_urls = []
    image_filenames = []
    image_keys = []

    for key in keys:
        start = time.time()
        try:
            output_folder = os.path.join(download_folder, key.replace(" ", ""))
            create_folder_if_it_doesnt_exist(output_folder)
            all_hits = []
            for page in range(1, num_pages + 1):
                result = pixabay_api_request(
                    api_key=api_key,
                    query=key,
                    lang="en",
                    min_width=min_width,
                    min_height=min_height,
                    order=order,
                    per_page=per_page,
                    page=page,
                )
                all_hits.extend(result["hits"])

            image_urls_for_this_key = [x["largeImageURL"] for x in all_hits]
            image_urls.extend(image_urls_for_this_key)
            image_keys.extend([key for x in all_hits])

            for index, url in enumerate(tqdm(image_urls_for_this_key, desc = f"Downloading images for key: {key}")):
                image_filename = os.path.join(output_folder, f"{index}.jpg")
                image = load_pil_image(path_or_url=url)
                # Optionally convert image to RGB if it is not
                if image.mode != "RGB":
                    image = image.convert("RGB")
                image.save(
                    image_filename
                )
            filenames_in_folder = get_filenames_in_a_folder(output_folder)
            image_filenames.extend(filenames_in_folder)

        except KeyboardInterrupt:
            print(f"Detected keyboard interrupt, stopping download...")
            break

        end = time.time()
        scraping_speed = len(filenames_in_folder) / (end - start)
        print(f"Scraping speed: {scraping_speed} images/second")

    data = {"key": image_keys, "filename": image_filenames, "url": image_urls}
    print(f"Total number of images: {len(data['filename'])}")
    save_as_json(data, output_json_file)
