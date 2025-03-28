import os
import shutil

# Input dataset path (original images)
input_dataset_path = r'D:\mehrs\VScode(projects)\miniProject\ChilliDataset(afterAug)'

# Output dataset path (renamed images)
output_dataset_path = r'D:\mehrs\VScode(projects)\miniProject\ChilliDataset(afterRename)'

# Ensure output directory exists
os.makedirs(output_dataset_path, exist_ok=True)

# Loop through each folder inside the dataset directory
for folder_name in os.listdir(input_dataset_path):
    folder_path = os.path.join(input_dataset_path, folder_name)

    # Check if it's a directory
    if not os.path.isdir(folder_path):
        continue

    # Create corresponding folder in output directory
    output_folder_path = os.path.join(output_dataset_path, folder_name)
    os.makedirs(output_folder_path, exist_ok=True)

    # List and sort files
    files = sorted(os.listdir(folder_path))

    # Loop through files and rename them
    for index, file_name in enumerate(files, start=1):
        file_path = os.path.join(folder_path, file_name)

        # Ensure it's a file (not a folder)
        if not os.path.isfile(file_path):
            continue

        # Get file extension
        file_extension = os.path.splitext(file_name)[1]

        # Create new file name in format 'subfoldername1.jpg', 'subfoldername2.jpg', ...
        new_file_name = f"{folder_name}{index}{file_extension}"
        new_file_path = os.path.join(output_folder_path, new_file_name)

        # Copy and rename file to the output directory
        shutil.copy(file_path, new_file_path)

    print(f"Renaming and saving completed for {folder_name}")

print("All files renamed and saved successfully!")
