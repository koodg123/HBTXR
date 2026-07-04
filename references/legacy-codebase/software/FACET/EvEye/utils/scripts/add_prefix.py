import os


def add_prefix_to_png_files(directory, prefix="right_"):
    # Iterate over all files and folders in the specified directory
    for filename in os.listdir(directory):
        # Check whether the file extension is .png
        if filename.endswith(".png"):
            # Build the full path of the original file
            old_file = os.path.join(directory, filename)
            # Build the new file name by prefixing the original file name with "left_"
            new_filename = prefix + filename
            # Build the full path of the new file
            new_file = os.path.join(directory, new_filename)
            # Rename the file
            os.rename(old_file, new_file)
            print(f"Renamed {old_file} to {new_file}")


def main():
    target_directory = "/mnt/data2T/junyuan/eye-tracking/datasets/Data_davis_labelled_with_mask/right/data"
    add_prefix_to_png_files(target_directory)


if __name__ == "__main__":
    main()
