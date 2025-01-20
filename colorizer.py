import os
import argparse
import torch
from model.eccv16 import eccv16
from model.util import load_img, preprocess_img, postprocess_tens
from torchvision.transforms import Grayscale
from PIL import Image
import numpy as np

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_dir', type=str, required=True, help='Path to the input image directory')
parser.add_argument('-o', '--output_dir', type=str, required=True, help='Path to the output directory')
args = parser.parse_args()

# Create output directories
os.makedirs(os.path.join(args.output_dir, "base_model"), exist_ok=True)
os.makedirs(os.path.join(args.output_dir, "fine_tune_model"), exist_ok=True)
os.makedirs(os.path.join(args.output_dir, "grayscale"), exist_ok=True)

# Load models
base_model = eccv16(pretrained=True, use_finetune_layer=False, use_training=False)
base_model.eval()

fine_tune_model = eccv16(pretrained=True, use_finetune_layer=True, use_training=False)
fine_tune_model.eval()

# Iterate through input images
for img_name in os.listdir(args.input_dir):
    img_path = os.path.join(args.input_dir, img_name)

    # Skip non-image files
    if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    # Load and preprocess image
    img = load_img(img_path)
    tens_l_orig, tens_l_rs = preprocess_img(img, HW=(256, 256))

    # Generate outputs
    with torch.no_grad():
        # Base model
        base_output = postprocess_tens(tens_l_orig, base_model(tens_l_rs).cpu())
        # Fine-tune model
        fine_tune_output = postprocess_tens(tens_l_orig, fine_tune_model(tens_l_rs).cpu())

    # Convert outputs to images
    base_output_img = Image.fromarray((base_output * 255).astype('uint8'))
    fine_tune_output_img = Image.fromarray((fine_tune_output * 255).astype('uint8'))


    # Save outputs
    base_output_path = os.path.join(args.output_dir, "base_model", f"{os.path.splitext(img_name)[0]}_base.png")
    fine_tune_output_path = os.path.join(args.output_dir, "fine_tune_model", f"{os.path.splitext(img_name)[0]}_fine_tune.png")


    base_output_img.save(base_output_path)
    fine_tune_output_img.save(fine_tune_output_path)


    print(f"Processed and saved: {img_name}")

print("Processing complete.")
