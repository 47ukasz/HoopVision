from pathlib import Path
import numpy as np
import shutil

project_dir = Path.cwd()
images_dir = project_dir / "data/collection/images"
labels_dir = project_dir / "data/collection/labels"

for split in ["train", "val"]:
    img_dir = project_dir / f"data/images/{split}"
    lbl_dir = project_dir / f"data/labels/{split}"

    if img_dir.exists():
        shutil.rmtree(img_dir)
    if lbl_dir.exists():
        shutil.rmtree(lbl_dir)

    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

images_file_list = list(images_dir.glob("*.jpg"))

if len(images_file_list) == 0:
    raise ValueError("Nie znaleziono żadnych obrazów.") 

shuffled_images_file_list = np.random.permutation(images_file_list)

validation_size = 0.3
validation_index = int(len(shuffled_images_file_list) * validation_size)

for index, filePath in enumerate(shuffled_images_file_list):
    dir_name = ""

    if index < validation_index:
        dir_name = "val"
    else:
        dir_name = "train"

    correspondLabelFile = labels_dir / f"{filePath.stem}.txt"

    destinationImageDir = project_dir / f"data/images/{dir_name}"
    destinationLabelDir = project_dir / f"data/labels/{dir_name}"

    shutil.copy(filePath, destinationImageDir) # kopiowanie obrazu
    shutil.copy(correspondLabelFile, destinationLabelDir) # kopiowanie labelu
