import albumentations as A
import cv2
import numpy as np
import os
import random
from tqdm import tqdm

# Configuration
INPUT_DATASET = r"D:\mehrs\VSCodeProjects\miniProject\DataSets\cropedDataset"
OUTPUT_DATASET = r"D:\mehrs\VSCodeProjects\miniProject\DataSets\Dataset(afterAUG)1"
TARGET_PER_CLASS = 1000
RANDOM_SEED = 42
IMAGE_SIZE = (260, 260)

def get_disease_specific_augmentation(class_name):
    """Returns augmentation pipeline tailored for specific chili diseases"""
    base_pipeline = [
        A.Resize(IMAGE_SIZE[0], IMAGE_SIZE[1], interpolation=cv2.INTER_CUBIC),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Rotate(limit=30, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5)
    ]
    
    # Disease-specific additions
    if 'BacterialSpots' in class_name:
        base_pipeline.extend([
            A.CLAHE(clip_limit=3.0, p=0.7),
            A.Spatter(p=0.4),  # Increased for spot simulation
            A.GaussianBlur(blur_limit=(2, 4), p=0.4),
            A.MultiplicativeNoise(multiplier=(0.8, 1.2), p=0.3)
        ])
    elif 'LeafCurl' in class_name:
        base_pipeline.extend([
            A.ElasticTransform(alpha=150, sigma=150*0.07, p=0.7),
            A.Perspective(scale=(0.07, 0.12), p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.4)
        ])
    elif 'Deficiency' in class_name:  # For both Magnesium and Zinc
        base_pipeline.extend([
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, p=0.7),
            A.RandomGamma(gamma_limit=(80, 120), p=0.5),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.4)
        ])
    else:  # Healthy leaves
        base_pipeline.extend([
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.RandomShadow(p=0.2)
        ])
    
    return A.Compose(base_pipeline)

def process_class(class_path, output_path, class_name):
    original_images = []
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    
    # Load and resize images
    for img_file in os.listdir(class_path):
        if img_file.lower().endswith(valid_extensions):
            img_path = os.path.join(class_path, img_file)
            image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_CUBIC)
            original_images.append(image)
    
    if not original_images:
        print(f"No images found in {class_path}")
        return

    # Get disease-specific augmentation
    augmentor = get_disease_specific_augmentation(class_name)
    original_count = len(original_images)
    augmented_images = []
    
    if original_count >= TARGET_PER_CLASS:
        combined = random.sample(original_images, TARGET_PER_CLASS)
    else:
        needed = TARGET_PER_CLASS - original_count
        base_aug = needed // original_count
        remainder = needed % original_count
        
        for idx, img in enumerate(original_images):
            num_aug = base_aug + 1 if idx < remainder else base_aug
            for _ in range(num_aug):
                augmented = augmentor(image=img)['image']
                
                # Post-processing for specific diseases
                if 'BacterialSpots' in class_name:
                    lab = cv2.cvtColor(augmented, cv2.COLOR_RGB2LAB)
                    l, a, b = cv2.split(lab)
                    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
                    l = clahe.apply(l)
                    augmented = cv2.cvtColor(cv2.merge((l,a,b)), cv2.COLOR_LAB2RGB)
                elif 'LeafCurl' in class_name:
                    gray = cv2.cvtColor(augmented, cv2.COLOR_RGB2GRAY)
                    edges = cv2.Canny(gray, 30, 100)
                    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
                    augmented = cv2.addWeighted(augmented, 0.85, edges, 0.15, 0)
                
                augmented_images.append(augmented)
        
        combined = original_images + augmented_images
    
    # Shuffle and save
    random.shuffle(combined)
    os.makedirs(output_path, exist_ok=True)
    
    for i, img in enumerate(combined[:TARGET_PER_CLASS]):
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(output_path, f"{class_name}_aug_{i:04d}.jpg"), img_bgr)

def main():
    random.seed(RANDOM_SEED)
    
    # Create identical folder structure
    for class_name in os.listdir(INPUT_DATASET):
        class_path = os.path.join(INPUT_DATASET, class_name)
        if os.path.isdir(class_path):
            os.makedirs(os.path.join(OUTPUT_DATASET, class_name), exist_ok=True)
    
    # Process each disease class
    class_dirs = [d for d in os.listdir(INPUT_DATASET) if os.path.isdir(os.path.join(INPUT_DATASET, d))]
    
    for class_name in tqdm(class_dirs, desc="Processing diseases"):
        class_path = os.path.join(INPUT_DATASET, class_name)
        output_path = os.path.join(OUTPUT_DATASET, class_name)
        process_class(class_path, output_path, class_name)

if __name__ == "__main__":
    main()
    print(f"Augmentation complete. Dataset saved to: {OUTPUT_DATASET}")