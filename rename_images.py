import os
import re

def rename_images(directory):
    files = os.listdir(directory)

    # Regular expression to match files starting with "Amostra "
    pattern = re.compile(r'^Amostra\s+(\d+)')

    for filename in files:
        # Check if the filename matches the pattern
        match = pattern.match(filename)
        if match:
            # Extract the number part from the filename
            number = match.group(1)

            # New filename without "Amostra "
            new_filename = f"a{number}.jpg"  # Change the extension if necessary

            # Rename the file
            os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))

if __name__ == "__main__":
    # Specify the directory containing the images
    directory = "./datasets/CD3/images"

    # Call the function to rename images
    rename_images(directory)
