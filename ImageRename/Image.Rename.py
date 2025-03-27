import os

# Define the dataset folder path
dataset_path = r'D:\mehrs\VScode(projects)\miniProject\ImageRename\ChilliDataset33'

# Loop through each folder inside the dataset directory
for folder_name in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder_name)

    # Check if it's a directory
    if not os.path.isdir(folder_path):
        continue

    # List all files in the folder
    files = os.listdir(folder_path)

    # Sort files to maintain order
    files.sort()

    # Loop through files and rename them
    for index, file_name in enumerate(files, start=1):
        file_path = os.path.join(folder_path, file_name)

        # Ensure it's a file (not a folder)
        if not os.path.isfile(file_path):
            continue

        # Get file extension
        file_extension = os.path.splitext(file_name)[1]

        # Create new file name in the format 'subfoldername1.jpg', 'subfoldername2.jpg', ...
        new_file_name = f"{folder_name}{index}{file_extension}"
        new_file_path = os.path.join(folder_path, new_file_name)

        # Rename file
        os.rename(file_path, new_file_path)

    print(f"Renaming completed for {folder_name}")

print("All files renamed successfully!")
