import os


def process_file(input_file):
    with open(input_file, "r") as file:
        lines = file.readlines()

    modified_lines = []
    last_valid_coords = None  # Stores the last valid coordinates
    future_coords_index = None  # Index used to search for coordinates after "No contour found"

    for i in range(len(lines)):
        line = lines[i]
        if "No contour found" in line:
            if last_valid_coords:
                # Replace with the previous valid coordinates
                new_line = f'{line.split(",")[0]},{last_valid_coords},1\n'
                modified_lines.append(new_line)
            else:
                # If no previous valid coordinates exist, search forward for valid coordinates
                future_coords_index = i + 1
                while future_coords_index < len(lines):
                    future_line = lines[future_coords_index]
                    parts = future_line.strip().split(",")
                    if len(parts) > 3:
                        # Found the first following valid coordinates
                        future_valid_coords = ",".join(parts[1:3])
                        new_line = f'{line.split(",")[0]},{future_valid_coords},1\n'
                        modified_lines.append(new_line)
                        break
                    future_coords_index += 1
                # If no valid coordinates are found in the whole file
                if future_coords_index >= len(lines):
                    modified_lines.append(line)
        else:
            parts = line.strip().split(",")
            if len(parts) > 3:
                last_valid_coords = ",".join(parts[1:3])
            modified_lines.append(line)

    # Write to the same file to overwrite the original data
    with open(input_file, "w") as file:
        file.writelines(modified_lines)


def process_all_txt_files(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            process_file(file_path)
            print(f"Processed and updated {file_path}")


def main():
    folder_path = "/mnt/data2T/junyuan/eye-tracking/datasets/DavisEyeCenterDataset/test/label"  # Replace with your folder path
    process_all_txt_files(folder_path)


if __name__ == "__main__":
    main()
