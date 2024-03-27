import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import os
import torch
import random
import numpy as np
import tifffile as tiff
from pathlib import Path
from omegaconf import OmegaConf
import importlib
import matplotlib.pyplot as plt
import torch.nn as nn

import sys
sys.path.append('latent_diffusion_main/')

from latent_diffusion_main.ldm.util import instantiate_from_config

def image_read(image_path):
    img = tiff.imread(image_path)
    img = (img / 1.0).transpose((2, 0, 1))

    image = torch.from_numpy((img.copy())).float()
    image = image / 10000.0
    mean = torch.as_tensor([0.5, 0.5, 0.5, 0.5],
                           dtype=image.dtype, device=image.device)
    std = torch.as_tensor([0.5, 0.5, 0.5, 0.5],
                          dtype=image.dtype, device=image.device)
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    image.sub_(mean).div_(std)

    return image

def get_rgb(image):
    image = image.mul(0.5).add_(0.5)
    image = image.squeeze()
    image = image.mul(10000).add_(0.5).clamp_(0, 10000)
    image = image.permute(1, 2, 0).cpu().detach().numpy()
    image = image.astype(np.uint16)

    r = image[:, :, 0]
    g = image[:, :, 1]
    b = image[:, :, 2]

    r = np.clip(r, 0, 2000)
    g = np.clip(g, 0, 2000)
    b = np.clip(b, 0, 2000)

    rgb = np.dstack((r, g, b))
    rgb = rgb - np.nanmin(rgb)

    if np.nanmax(rgb) == 0:
        rgb = 255 * np.ones_like(rgb)
    else:
        rgb = 255 * (rgb / np.nanmax(rgb))

    rgb[np.isnan(rgb)] = np.nanmean(rgb)
    rgb = rgb.astype(np.float32)

    return rgb / 255.0

# directory_path = "D:/drive/research_uofa/LDM_exps/CTGAN/CTGAN/Sen2_MTC/dataset/Sen2_MTC/T09WWP_R071/cloudless/"
# image_name = "T09WWP_R071_0.tif"

# im_original = image_read_custom(directory_path + "/" + image_name)[:, :, 0:3]
# im = im_original.reshape(1, 3, 256, 256)
# im_original = image_read(directory_path + "/" + image_name)

image_list = Path("Sen2_MTC_New_Multi/dataset/Sen2_MTC/").glob("**/*.tif")

def encode_image(img_path, model):
    im_original = image_read(img_path)
    im_original = get_rgb(im_original)
    im = torch.from_numpy(np.array(im_original, dtype=np.float32)).to("cuda:0")
    im = im.permute(2, 0, 1)
    im = im.reshape(1, 3, 256, 256)
    q, l, info = model.encode(im)
    return q, im_original


# image_list = [i for i in list(image_list)]
path_conf = "latent_diffusion_main/models/first_stage_models/vq-f4/config.yaml"
config = OmegaConf.load(path_conf)
model = instantiate_from_config(config.model)
model.load_state_dict(torch.load("../model.ckpt")['state_dict'], strict=False)
model = model.to("cuda:0")

# image_list = ['Sen2_MTC_New_Multi/dataset/Sen2_MTC/T41VMJ_R006/cloudless/T41VMJ_R006_0.tif']
image_list = ['Latent_MTC_New/dataset/Sen2_MTC/T41VMJ_R006/cloudless/T41VMJ_R006_0.npy']

for img_path in image_list:
    image = np.load(str(img_path))
    # encoded_image, im_original = encode_image(image, model)
    # print(encoded_image.shape)
    # exit()
    print(image.shape)
    encoded_image = torch.from_numpy(np.array(image, dtype=np.float32)).to("cuda:0")
    restored = model.decode(encoded_image)
    restored = restored.cpu()
    restored = restored.detach().numpy().transpose(0, 2, 3, 1)[0]
    # print(restored.shape)

    # im_original = im_original.cpu()
    # im_original = im_original.transpose(0, 2, 3, 1)][0]

    # plt.subplot(1,2,1)
    # # im_original = im_original.reshape(256, 256, 3).cpu()
    # plt.imshow(im_original)
    plt.subplot(1,2,2)
    # restored = restored.reshape(256, 256, 3)
    plt.imshow(restored)
    plt.show()
    break

# img_path_1 = 'cjen_images/GT_T41VMJ_R006_0.npy'
img_path_1 = np.load('GT_good_example.npy')
img_path_2 = np.load('Pred_good_example.npy')
print(img_path_1.shape, img_path_2.shape)
# exit()

# # img_path_2 = 'cjen_images/Out_T41VMJ_R006_0.npy'
# # img_path_2 = 'cjen_images/T41VMJ_R006_0_1.npy'
# image_1 = np.load(str(img_path_1))
# # image_1 = image_1.transpose(2, 0, 1)
# image_1 = image_1.reshape(1, 3, 64, 64)
# print(image_1.shape)
# image_2 = np.load(str(img_path_2))
# # image_2 = image_2.transpose(2, 0, 1)
# image_2 = image_2.reshape(1, 3, 64, 64)
# # encoded_image, im_original = encode_image(image, model)
# # print(encoded_image.shape)
# # exit()

image_1 = img_path_1[3, :, :, :]
image_1 = image_1.reshape(1, 3, 64, 64)
image_2 = img_path_2[3, :, :, :]
image_2 = image_2.reshape(1, 3, 64, 64)

encoded_image_1 = torch.from_numpy(np.array(image_1, dtype=np.float32)).to("cuda:0")
restored_1 = model.decode(encoded_image_1)
t1 = restored_1.clone()
restored_1 = restored_1.cpu()
restored_1 = restored_1.detach().numpy().transpose(0, 2, 3, 1)[0]

# print(restored_1.shape)

encoded_image_2 = torch.from_numpy(np.array(image_2, dtype=np.float32)).to("cuda:0")
restored_2 = model.decode(encoded_image_2)
t2 = restored_2.clone()
restored_2 = restored_2.cpu()
restored_2 = restored_2.detach().numpy().transpose(0, 2, 3, 1)[0]


criterion = nn.MSELoss()
print(criterion(encoded_image_1, encoded_image_2))  

plt.subplot(1,2,1)
plt.imshow(restored_1)
plt.subplot(1,2,2)
plt.imshow(restored_2)
plt.show()

# print(restored_1.shape)