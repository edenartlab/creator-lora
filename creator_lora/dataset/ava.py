"""
original dataset link:
https://academictorrents.com/details/71631f83b11d3d79d8f84efe0a7e12f0ac001460
"""
import os
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

def build_ava_dataset(
    download_folder: str,
    extract_image_7z_files: bool = False
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