import os
import urllib.request

import cv2


def download_file(url: str, save_dir: str = "./data") -> str:
    os.makedirs(save_dir, exist_ok=True)
    file_name = os.path.basename(url)
    save_path = os.path.join(save_dir, file_name)

    with urllib.request.urlopen(url) as web_file:
        with open(save_path, mode="wb") as local_file:
            local_file.write(web_file.read())

    return save_path


def play_gif(file_name: str):
    # GIF動画を読み込む
    gif = cv2.VideoCapture(file_name)
    # フレームを配列に格納する
    frames = []
    while True:
        ret, frame = gif.read()
        if not ret:
            break
        frames.append(frame)
    # フレームをループして表示する
    while True:
        for frame in frames:
            cv2.imshow("GIF", frame)
            if cv2.waitKey(100) & 0xFF == 27:
                break
