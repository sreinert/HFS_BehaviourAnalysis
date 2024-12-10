import os
import shutil

def copy_most_recent_folder(source_folder, target_folder):
    folders = os.listdir(source_folder)
    if not folders:
        print("No folders found in the source directory.")
        return

    most_recent_folder = max(folders, key=lambda f: os.path.getmtime(os.path.join(source_folder, f)))
    source_path = os.path.join(source_folder, most_recent_folder)
    target_path = os.path.join(target_folder, most_recent_folder)

    try:
        shutil.copytree(source_path, target_path)
        print(f"Successfully copied the most recent folder '{most_recent_folder}' to the target location.")
    except Exception as e:
        print(f"An error occurred while copying the folder: {e}")

#now write a function that iterates over a list of source folders and and runs the function above
def grab_training_files(source_folders, target_folders):
    for source_folder in source_folders:
        target_folder = target_folders[source_folders.index(source_folder)]
        copy_most_recent_folder(source_folder, target_folder)


# Example usage
base_folder = "/path/to/base/folder"
source_folders = [os.path.join(base_folder, "source1"), os.path.join(base_folder, "source2")]
base_target_folder = "/path/to/target/folder"
target_folders = [os.path.join(base_target_folder, "target1"), os.path.join(base_target_folder, "target2")]

grab_training_files(source_folders, target_folders)

