import os
import cv2
import numpy as np
import albumentations as A

# Fixed image size (change as needed)
FIXED_SIZE = (512, 512)  # Keep all images 512x512


transform = A.Compose([
    A.RandomRotate90(),  # Random 90-degree rotations
    A.HorizontalFlip(p=0.5),  # Flip horizontally
    A.VerticalFlip(p=0.5),  # Flip vertically
    A.RandomBrightnessContrast(p=0.2),  # Adjust brightness and contrast
    A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=20, val_shift_limit=15, p=0.3),  # Color jitter
    A.MotionBlur(blur_limit=3, p=0.2),  # Motion blur
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),  # Gaussian blur
    A.GaussNoise(var_limit=(5.0, 20.0), p=0.3),  # Reduced Gaussian noise
    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),  # Elastic deformation
])

def resize_with_padding(image, size=FIXED_SIZE):
    """Resizes image while maintaining aspect ratio by adding padding."""
    h, w = image.shape[:2]
    scale = min(size[0] / w, size[1] / h)
    
    # Resize image
    new_w, new_h = int(w * scale), int(h * scale)
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create new image with padding
    padded_image = np.full((size[1], size[0], 3), 0, dtype=np.uint8)  # White background
    pad_x = (size[0] - new_w) // 2
    pad_y = (size[1] - new_h) // 2
    padded_image[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized_image

    return padded_image

def augment_and_save_image(image_path, output_path, num_augmented=5):
    """Loads an image, resizes it, applies augmentation, and saves multiple augmented versions."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image {image_path}")
        return

    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize to fixed size while maintaining aspect ratio
    image = resize_with_padding(image, FIXED_SIZE)

    for i in range(num_augmented):
        augmented = transform(image=image)['image']  # Apply augmentation

        # Convert back to BGR for saving
        augmented = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)

        # Create unique output filename
        base_name, ext = os.path.splitext(os.path.basename(image_path))
        augmented_filename = f"{base_name}_aug{i+1}{ext}"
        augmented_output_path = os.path.join(output_path, augmented_filename)

        cv2.imwrite(augmented_output_path, augmented)
        print(f"Saved: {augmented_output_path}")

def process_images(input_folder, output_folder, num_augmented=4):
    """Processes all images in the input folder and saves augmented images."""
    os.makedirs(output_folder, exist_ok=True)

    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                input_path = os.path.join(root, file)
                
                # Maintain folder structure
                relative_path = os.path.relpath(root, input_folder)
                output_dir = os.path.join(output_folder, relative_path)
                os.makedirs(output_dir, exist_ok=True)

                augment_and_save_image(input_path, output_dir, num_augmented)

# Paths
input_folder = r"D:\mehrs\VScode(projects)\miniProject\ChilliDataset"
output_folder = r"D:\mehrs\VScode(projects)\miniProject\ChilliDataset(afterAug)"

# Process images with augmentation
process_images(input_folder, output_folder, num_augmented=4)
print("Augmentation complete! 5 versions generated per image.")
