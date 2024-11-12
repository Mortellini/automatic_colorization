
from palette import *
from palette.color_counter import count_colors_in_directory
import torch
import numpy as np
#model = torch.load('colorization_model_h_new.pth')
#print(model)
#count_colors_in_directory("./images", "palette/colors.txt", 10)

# Load the pts_in_hull.npy file
pts_in_hull = np.load('pts_in_hull.npy')

# Print the shape and content of pts_in_hull
print(f'Shape of pts_in_hull: {pts_in_hull.shape}')
print(f'Content of pts_in_hull:\n{pts_in_hull}')