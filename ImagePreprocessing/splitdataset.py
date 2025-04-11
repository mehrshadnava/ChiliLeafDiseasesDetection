import os
import shutil
import random

original_dataset = r"D:\mehrs\VSCodeProjects\miniProject\DataSets\dataset3k"
output_dir = r"D:\mehrs\VSCodeProjects\miniProject\DataSets\split_dataset"

train_ratio = 0.7
valid_ratio = 0.15
test_ratio = 0.15

classes = os.listdir(original_dataset)

for cls in classes:
    cls_path = os.path.join(original_dataset, cls)
    images = os.listdir(cls_path)
    random.shuffle(images)

    total = len(images)
    train_end = int(train_ratio * total)
    valid_end = int((train_ratio + valid_ratio) * total)

    splits = {
        "train": images[:train_end],
        "valid": images[train_end:valid_end],
        "test": images[valid_end:]
    }

    for split, split_imgs in splits.items():
        split_dir = os.path.join(output_dir, split, cls)
        os.makedirs(split_dir, exist_ok=True)
        for img in split_imgs:
            src = os.path.join(cls_path, img)
            dst = os.path.join(split_dir, img)
            shutil.copy(src, dst)

print("Dataset split complete!")
