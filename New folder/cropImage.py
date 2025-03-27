import cv2
import numpy as np
import os

def crop_and_resize_image(image_path, output_path, output_size=(256, 256)):
    # Load the image
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Unable to load image {image_path}")
        return

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the image to create a binary mask
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    # Find contours of the leaf
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Get the bounding box of the largest contour (assuming it's the leaf)
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)

        print(f"Bounding box of leaf: x={x}, y={y}, w={w}, h={h}")

        # Crop the image to the bounding box
        cropped_image = image[y:y+h, x:x+w]

        # Calculate scaling ratio to fit the leaf into the 256x256 size
        crop_h, crop_w = cropped_image.shape[:2]
        scale = min(output_size[0] / crop_w, output_size[1] / crop_h)

        # Resize the cropped image while maintaining aspect ratio
        resized_image = cv2.resize(cropped_image, (int(crop_w * scale), int(crop_h * scale)))

        # Create a blank canvas of the desired output size
        canvas = np.ones((output_size[1], output_size[0], 3), dtype=np.uint8) * 255  # White canvas

        # Calculate the center position to place the resized image
        start_x = (output_size[0] - resized_image.shape[1]) // 2
        start_y = (output_size[1] - resized_image.shape[0]) // 2

        # Place the resized image onto the canvas
        canvas[start_y:start_y + resized_image.shape[0], start_x:start_x + resized_image.shape[1]] = resized_image

        # Save the final image
        cv2.imwrite(output_path, canvas)

        print(f"Cropped and resized image saved to {output_path}")
    else:
        print(f"No leaf detected in the image: {image_path}")

# Recursively process all images in subfolders
def process_images_in_folders(input_folder, output_folder, output_size=(256, 256)):
    # Loop through the files and subfolders in the input folder
    for root, dirs, files in os.walk(input_folder):
        # Create corresponding output subfolders
        for dir_name in dirs:
            output_subfolder = os.path.join(output_folder, os.path.relpath(os.path.join(root, dir_name), input_folder))
            if not os.path.exists(output_subfolder):
                os.makedirs(output_subfolder)

        # Process image files in the current folder
        for file_name in files:
            if file_name.endswith(('.jpg', '.jpeg', '.png')):  # Add other image formats if necessary
                input_image_path = os.path.join(root, file_name)
                relative_path = os.path.relpath(input_image_path, input_folder)
                output_image_path = os.path.join(output_folder, relative_path)

                # Ensure the output directory exists
                output_dir = os.path.dirname(output_image_path)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                # Crop, resize, and save the image
                crop_and_resize_image(input_image_path, output_image_path, output_size)

input_folder = r'D:\mehrs\VScode(projects)\miniProject\New folder\ChilliDataset'
output_folder = r'D:\mehrs\VScode(projects)\miniProject\New folder\ChilliDataSet1'

# Ensure output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Process all images recursively
process_images_in_folders(input_folder, output_folder)

print("All images processed!")
