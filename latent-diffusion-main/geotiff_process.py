from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import tifffile as tiff

directory_path = "D:/drive/research_uofa/LDM_exps/CTGAN/CTGAN/Sen2_MTC/dataset/Sen2_MTC/T21MUR_R010/cloudless/"
image_name = "T21MUR_R010_51.tif"




def image_read_custom(image_path):
    img = tiff.imread(image_path)
    img = (img / 1.0)

    image = torch.from_numpy((img.copy())).float()
    image = image / 10000.0

    return image.reshape(256, 256, 4)


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


# im = image_read(directory_path + "/" + image_name)
# im = get_rgb(im)
# plt.imshow(im)
# plt.show()