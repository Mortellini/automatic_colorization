from PIL import Image  # Python Imaging Library for image processing tasks
import numpy as np  # NumPy for numerical computations
from skimage import color  # scikit-image's color module for color space conversions
import torch  # PyTorch for tensor operations
import torch.nn.functional as F  # PyTorch's functional module for operations like interpolation
from scipy.spatial import cKDTree
from IPython import embed  # IPython's embed function for debugging (interactive shell)

# Load pts_in_hull file (contains the 313 quantized ab values)
pts_in_hull = np.load('pts_in_hull.npy')

# Create KDTree for fast nearest-neighbor lookup
pts_tree = cKDTree(pts_in_hull)


# Function to load an image from a given path and ensure it's in RGB format
def load_img(img_path):
    # Convert image to a NumPy array
    out_np = np.asarray(Image.open(img_path))

    # If the image is grayscale (2D array), convert it to 3D by replicating the grayscale channel across RGB
    if (out_np.ndim == 2):
        out_np = np.tile(out_np[:, :, None], 3)

    return out_np  # Return the image as a NumPy array


# Function to resize an image to the specified dimensions (default 256x256) using a specified resampling method
def resize_img(img, HW=(256, 256), resample=3):
    # Convert the image array back to a PIL Image, resize it, and convert back to a NumPy array
    return np.asarray(Image.fromarray(img).resize((HW[1], HW[0]), resample=resample))


# Function to preprocess an image for input to a neural network
def preprocess_img(img_rgb_orig, HW=(256, 256), resample=3):
    # Resize the original RGB image to the specified dimensions
    img_rgb_rs = resize_img(img_rgb_orig, HW=HW, resample=resample)

    # Convert the original and resized RGB images to LAB color space
    img_lab_orig = color.rgb2lab(img_rgb_orig)
    img_lab_rs = color.rgb2lab(img_rgb_rs)

    # Extract the L channel (lightness) from both the original and resized LAB images
    img_l_orig = img_lab_orig[:, :, 0]
    img_l_rs = img_lab_rs[:, :, 0]

    # Convert the L channels to PyTorch tensors and add batch and channel dimensions
    tens_orig_l = torch.Tensor(img_l_orig)[None, None, :, :]
    tens_rs_l = torch.Tensor(img_l_rs)[None, None, :, :]

    # Return both the original size L tensor and the resized L tensor
    return (tens_orig_l, tens_rs_l)

# TODO make it possible to turn resize off and on
# Function to postprocess the output from a neural network back into an RGB image
def postprocess_tens(tens_orig_l, out_ab, mode='bilinear'):
    # tens_orig_l: the original L channel tensor (1 x 1 x H_orig x W_orig)
    # out_ab: the predicted AB channels tensor (1 x 2 x H x W)

    HW_orig = tens_orig_l.shape[2:]  # Original image height and width
    #print(HW_orig)
    HW = out_ab.shape[2:]  # Output image height and width
    #print(HW)

    # If the dimensions of the AB channels don't match the original L channel, resize the AB channels
    if (HW_orig[0] != HW[0] or HW_orig[1] != HW[1]):
        out_ab_orig = F.interpolate(out_ab, size=HW_orig, mode=mode)
        #print("ab channels " + str(out_ab_orig.shape))
    else:
        out_ab_orig = out_ab
        #print("ab channeles " + str(out_ab_orig.shape))

    # Concatenate the original L channel with the resized AB channels to form a complete LAB image
    out_lab_orig = torch.cat((tens_orig_l, out_ab_orig), dim=1)
    #print("LAB VALUES " + str(out_lab_orig.shape))

    # Convert the LAB image back to RGB and return it as a NumPy array
    return color.lab2rgb(out_lab_orig.data.cpu().numpy()[0, ...].transpose((1, 2, 0)))


def quantize_ab(ab_channels):
    """
    Quantize continuous AB values to the nearest color bin.

   Returns:
    - quantized_bins: Tensor of shape (batch_size, height, width), containing the indices (0-312) of the closest color bin.
    """
    h, w, _ = ab_channels.shape
    ab_quantized = np.zeros((h, w), dtype=np.int32)

    for i in range(h):
        for j in range(w):
            ab_val = ab_channels[i, j, :]
            _, idx = pts_tree.query(ab_val)  # Find the closest quantized bin
            ab_quantized[i, j] = idx  # Assign index of nearest bin

    return ab_quantized


def dequantize_ab(quantized_ab):
    """
    Convert predicted bin indices back to AB values.

    Returns:
    - predicted_ab: Tensor of shape (batch_size, 2, height, width), containing the predicted AB values.
    """
    h, w = quantized_ab.shape
    ab_decoded = np.zeros((h, w, 2), dtype=np.float32)

    for i in range(h):
        for j in range(w):
            idx = quantized_ab[i, j]
            ab_decoded[i, j, :] = pts_in_hull[idx, :]  # Get actual ab values from bin index

    return ab_decoded





