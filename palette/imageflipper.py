import os
from PIL import Image, ExifTags


# Function to correct the orientation of a single image
def correct_image_orientation(image_path):
    image = Image.open(image_path)

    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = image._getexif()

        if exif is not None:
            exif_orientation = exif.get(orientation)
            if exif_orientation == 3:
                image = image.rotate(180, expand=True)
            elif exif_orientation == 6:
                image = image.rotate(270, expand=True)
            elif exif_orientation == 8:
                image = image.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        # No EXIF orientation data
        pass

    return image


# Function to process all images in a directory
def correct_images_in_directory(directory_path, output_directory=None):
    # If output_directory is not provided, overwrite the images in the original directory
    if output_directory is None:
        output_directory = directory_path

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Iterate through all files in the directory
    for filename in os.listdir(directory_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):  # You can add more extensions if necessary
            file_path = os.path.join(directory_path, filename)

            # Correct orientation
            corrected_image = correct_image_orientation(file_path)

            # Save the corrected image in the output directory
            save_path = os.path.join(output_directory, filename)
            corrected_image.save(save_path)
            print(f"Processed and saved: {save_path}")


# Example usage:
directory_path = 'images/'
output_directory = 'images/'
correct_images_in_directory(directory_path, output_directory)
