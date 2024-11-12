from torchvision import transforms
from PIL import Image
import torch
import os
import numpy as np
import cv2


class ColorizationDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None, size=(256, 256)):
        self.root_dir = ""
        #self.file_names = os.listdir(root_dir)  # Assuming images are directly in root_dir

        self.file_names = []
        # Use os.walk to get all image file paths in the root directory and its subdirectories
        for dirpath, _, files in os.walk(root_dir):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):  # Filter image files
                    # Ensure that the full path is correct
                    #print(dirpath)
                    #print(f)
                    self.file_names.append(os.path.join(dirpath, f))

        self.transform = transform
        self.size = size  # Target size to resize images

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.file_names[idx])
        image = Image.open(img_name)

        # Ensure images are RGB (or grayscale if you're handling that)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Resize the image to the fixed size
        image = image.resize(self.size)

        # Apply any additional transformations (e.g., normalization)
        if self.transform:
            image = self.transform(image)

        # Convert the image to a tensor
        image_tensor = transforms.ToTensor()(image)

        # For simplicity, return the L channel and placeholder AB channels
        img_l = torch.unsqueeze(image_tensor[0, :, :], 0)  # L channel (grayscale)
        img_ab = torch.stack((image_tensor[1, :, :], image_tensor[2, :, :]), dim=0)  # AB channels

        return img_l, img_ab  # Return L as input, AB as target


# Function to convert LAB to RGB
def lab_to_rgb(L, ab):
    """Convert a LAB image to RGB. The input should be a batch of LAB images."""
    # Concatenate L with ab to form LAB image
    L = L * 100  # Rescale to match the LAB format
    ab = ab * 128  # Rescale to match the LAB format

    lab = torch.cat([L, ab], dim=1)  # Concatenate along the channel dimension

    # Convert LAB to RGB using OpenCV
    rgb_images = []
    for img in lab:
        img = img.detach().cpu().numpy().transpose(1, 2, 0).astype(np.float32)  # Detach from computation graph
        img_rgb = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
        rgb_images.append(img_rgb)

    # Stack the RGB images back into a batch
    rgb_images = np.stack(rgb_images, axis=0)
    rgb_images = torch.from_numpy(rgb_images).permute(0, 3, 1, 2)  # Convert to tensor and permute dimensions
    return rgb_images