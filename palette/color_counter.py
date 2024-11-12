import os
from collections import Counter
from PIL import Image

def is_grayish(color, threshold=30):
    """ Returns True if the color is close to gray (where R, G, and B are similar) """
    return abs(color[0] - color[1]) < threshold and abs(color[1] - color[2]) < threshold and abs(color[0] - color[2]) < threshold

def is_too_light_or_dark(color, light_threshold=240, dark_threshold=15):
    """ Exclude colors too close to white or black """
    avg = sum(color) // 3
    return avg >= light_threshold or avg <= dark_threshold

def count_colors_in_directory(directory_path, output_file, bin_size=10, gray_threshold=30, light_threshold=240, dark_threshold=15):
    # Counter for all the colors in all images
    color_counter = Counter()

    # Iterate over all files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_path = os.path.join(directory_path, filename)

            # Open the image
            with Image.open(image_path) as img:
                # Convert the image to RGB (if not already in RGB)
                img = img.convert('RGB')

                # Get the pixels of the image
                pixels = img.getdata()

                # Count occurrences of each color with binning applied
                for pixel in pixels:
                    # Apply color binning
                    binned_color = round_color(pixel, bin_size)

                    # Skip grayscale-like colors and those too close to black or white
                    if is_grayish(binned_color, gray_threshold) or is_too_light_or_dark(binned_color, light_threshold, dark_threshold):
                        continue

                    color_counter[binned_color] += 1

    # Get the most common colors
    most_common_colors = color_counter.most_common(100)

    # Write the result to the output file
    with open(output_file, 'w') as f:
        for color, count in most_common_colors:
            f.write(f"Color: {color}, Count: {count}\n")

    print(f"Top 50 colors (excluding gray, light, and dark colors) have been written to {output_file}")

def round_color(color, bin_size=10):
    """ Round each RGB value to the nearest multiple of bin_size """
    return tuple((c // bin_size) * bin_size for c in color)

# Example usage
# directory_path = "path/to/your/directory"
# output_file = "most_common_colors.txt"
# count_colors_in_directory(directory_path, output_file, bin_size=10, gray_threshold=30, light_threshold=240, dark_threshold=15)
