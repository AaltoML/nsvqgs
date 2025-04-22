"""Script to download benchmark dataset(s)"""

import os
import subprocess
from pathlib import Path
from typing import Literal
# # dataset names
dataset_names = Literal[
    "mipnerf360",
    "tandt_db"
]

class DownloadData:
    def __init__(self, save_dir: Path = Path(os.getcwd() + "/data")):
        self.save_dir = save_dir
        # dataset urls
        self.urls = {
            "mipnerf360": [
                "http://storage.googleapis.com/gresearch/refraw360/360_v2.zip",
                "https://storage.googleapis.com/gresearch/refraw360/360_extra_scenes.zip"
            ],
            # Downloading deep_blending, tanks_and_templs, refer to the website: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
            "tandt_db":[
                "https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip"
                ]
        }
        # rename maps
        self.dataset_rename_map = {
        "mipnerf360": "mipnerf360",
        "tandt_db": "."
        }

    def main(self):
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_download(self.dataset)

    def dataset_download(self, dataset:  dataset_names = "mipnerf360"):
        urls, dataset_rename_map = self.urls, self.dataset_rename_map
        if isinstance(urls[dataset], list):
            for url in urls[dataset]:
                url_file_name = Path(url).name
                extract_path = self.save_dir / dataset_rename_map[dataset]
                download_path = extract_path / url_file_name
                download_and_extract(url, download_path, extract_path)
        else:
            url = urls[dataset]
            url_file_name = Path(url).name
            extract_path = self.save_dir / dataset_rename_map[dataset]
            download_path = extract_path / url_file_name
            download_and_extract(url, download_path, extract_path)


def download_and_extract(url: str, download_path: Path, extract_path: Path) -> None:
    download_path.parent.mkdir(parents=True, exist_ok=True)
    extract_path.mkdir(parents=True, exist_ok=True)

    # download
    download_command = [
        "curl",
        "-L",
        "-o",
        str(download_path),
        url,
    ]
    try:
        subprocess.run(download_command, check=True)
        print("File file downloaded succesfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading file: {e}")

    # if .zip
    if Path(url).suffix == ".zip":
        if os.name == "nt":  # Windows doesn't have 'unzip' but 'tar' works
            extract_command = [
                "tar",
                "-xvf",
                download_path,
                "-C",
                extract_path,
            ]
        else:
            extract_command = [
                "unzip",
                download_path,
                "-d",
                extract_path,
            ]
    # if .tar
    else:
        extract_command = [
            "tar",
            "-xvzf",
            download_path,
            "-C",
            extract_path,
        ]

    # extract
    try:
        subprocess.run(extract_command, check=True)
        os.remove(download_path)
        print("Extraction complete.")
    except subprocess.CalledProcessError as e:
        print(f"Extraction failed: {e}")

# data Download
if __name__ == '__main__':
    
    save_dir = Path(os.getcwd() + "/data")
    dd = DownloadData(save_dir)
    print(save_dir)
    # # download MipNeRF360 dataset
    # ds = "mipnerf360"
    # if not os.path.isdir(save_dir/ds):
    #     print('Downloading [MipNeRF360] now...')
    #     dd.dataset_download(ds)
    
    # download "Tank and Template" and "Deep Blending" datasets
    ds = "tandt"
    if not (os.path.isdir(save_dir/ds) and os.path.isdir(f"data/db")):
        print('Downloading [Tank and Template] and [Deep Blending] now...')
        dd.dataset_download("tandt_db")