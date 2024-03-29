import os
import pexelsPy
import requests
from typing import List
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from ..utils.json_stuff import save_as_json, load_json

def build_pexels_dataset_json(
    pexels_api_key: str,
    keys: List[str],
    output_json_filename: str,
    results_per_page: int = 10,
    page=1,
):
    api = pexelsPy.API(pexels_api_key)

    data = {
        "id": [],
        "url": [],
        "duration": [],
        "width": [],
        "height": [],
        "filename": [],
        "key": [],
    }

    for key in tqdm(keys, desc = "Fetching video urls"):
        api.search_videos(key, page=page, results_per_page=results_per_page)
        videos = api.get_videos()

        data["id"].extend(
            [
                vid.id
                for vid in videos
            ]
        )
        
        data["url"].extend(
            [
                'https://www.pexels.com/video/' + str(vid.id) + '/download'
                for vid in videos
            ]
        )
        data["duration"].extend(
            [
                vid.duration
                for vid in videos
            ]
        )
        data["width"].extend(
            [
                vid.width
                for vid in videos
            ]
        )
        data["height"].extend(
            [
                vid.height
                for vid in videos
            ]
        )
        ## set None as placeholder
        data["filename"].extend(
            [
                None
                for vid in videos
            ]
        )
        data["key"].extend(
            [
                key
                for vid in videos
            ]
        )
    save_as_json(
        data,
        filename = output_json_filename
    )

def download_video(key, url, video_id, output_folder):
    filename = os.path.join(output_folder, f"key_{key.replace(' ', '')}_id_{video_id}.mp4")
    
    if not os.path.exists(filename):
        r = requests.get(url)
        with open(filename, 'wb') as outfile:
            outfile.write(r.content)

    return filename

def download_videos_from_dataset_json(output_json_filename, output_folder, num_threads=4):
    assert os.path.exists(output_json_filename)
    assert os.path.exists(output_folder)

    data = load_json(output_json_filename)

    all_urls = data["url"]
    all_keys = data["key"]
    all_ids = data["id"]

    filenames = []
    print(f"Will download {len(all_urls)} videos")

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for key, url, video_id in zip(all_keys, all_urls, all_ids):
            future = executor.submit(download_video, key, url, video_id, output_folder)
            futures.append(future)

        for future, key, url, video_id in zip(tqdm(futures, total=len(all_urls)), all_keys, all_urls, all_ids):
            filename = future.result()
            filenames.append(filename)

    data["filename"] = filenames
    save_as_json(data, filename=output_json_filename)