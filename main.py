import argparse
import matplotlib as plt
import matplotlib.pyplot as pyplot

from model import *
from model.eccv16 import eccv16
from model.util import *

parser = argparse.ArgumentParser()
parser.add_argument('-i','--img_path', type=str, default='images/latex/1.jpg')
# parser.add_argument('--use_gpu', action='store_true', help='whether to use GPU')
parser.add_argument('-o','--save_prefix', type=str, default='saved', help='will save into this file with {eccv16.png} suffix')
opt = parser.parse_args()

# load colorizers
colorizer_eccv16 = eccv16(pretrained=True, use_finetune_layer=True, use_training=False)
colorizer_eccv16.eval() # Switch the model to evaluation mode

# default size to process images is 256x256
# grab L channel in both original ("orig") and resized ("rs") resolutions
img = load_img(opt.img_path)
(tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256,256))

# colorizer outputs 256x256 ab map
# resize and concatenate to original L channel
img_bw = postprocess_tens(tens_l_orig, torch.cat((0*tens_l_orig,0*tens_l_orig),dim=1))
out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())

pyplot.imsave("", out_img_eccv16)


