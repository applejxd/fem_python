import os
import urllib.request


def download_file(url: str, save_dir: str = "./data") -> str:
    os.makedirs(save_dir, exist_ok=True)
    file_name = os.path.basename(url)
    save_path = os.path.join(save_dir, file_name)

    with urllib.request.urlopen(url) as web_file:
        with open(save_path, mode="wb") as local_file:
            local_file.write(web_file.read())

    return save_path
