import os
import cv2
import numpy as np
import albumentations as A

# Define augmentation sequence
augmentations = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.Rotate(limit=25, p=0.7, border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255)),  # Rotate with white background
    A.GaussianBlur(blur_limit=(3, 7), p=0.5),  # Apply Gaussian Blur
    A.RandomBrightnessContrast(p=0.5),  # Adjust brightness and contrast
])

# Define dataset path
dataset_path = r'D:\mehrs\VScode(projects)\miniProject\DataAugment\ChilliDataset11'

# Process each category folder
for category in os.scandir(dataset_path):
    if not category.is_dir():
        continue  # Skip if not a directory

    category_path = category.path
    print(f"Processing category: {category.name}")

    # Process each image in the folder
    for file in os.scandir(category_path):
        if not file.is_file() or not file.name.lower().endswith((".png", ".jpg", ".jpeg")):
            continue  # Skip non-image files

        image_path = file.path
        image = cv2.imread(image_path)

        if image is None:
            print(f"Skipping {file.name}, unable to read file.")
            continue

        base_name, ext = os.path.splitext(file.name)

        # Step 1: Convert the image background to white
        h, w, _ = image.shape
        white_background = np.full((h, w, 3), (255, 255, 255), dtype=np.uint8)

        # Find the mask of the non-white areas
        mask = image[:, :, :3] < 255
        white_background[mask] = image[mask]  # Only overwrite where the original image is not white

        # Step 2: Perform augmentation
        for i in range(1, 6):
            augmented = augmentations(image=white_background)["image"]

            new_image_name = f"{base_name}_aug{i}{ext}"
            new_image_path = os.path.join(category_path, new_image_name)

            try:
                cv2.imwrite(new_image_path, augmented)
            except Exception as e:
                print(f"Error saving {new_image_name}: {e}")

    print(f"Augmentation completed for {category.name}.\n")

print("Data augmentation completed for all folders.")
