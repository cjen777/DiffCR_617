import torch
import json
from PIL import Image
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
import numpy as np
import matplotlib.pyplot as plt
from geotiff_process import image_read_custom, image_read, get_rgb


def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    global_step = pl_sd["global_step"]
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()

    return model


directory_path = "D:/drive/research_uofa/LDM_exps/CTGAN/CTGAN/Sen2_MTC/dataset/Sen2_MTC/T09WWP_R071/cloudless/"
image_name = "T09WWP_R071_0.tif"

# im_original = image_read_custom(directory_path + "/" + image_name)[:, :, 0:3]
# im = im_original.reshape(1, 3, 256, 256)
im_original = image_read(directory_path + "/" + image_name)
im_original = get_rgb(im_original)

im = im_original.reshape(1, 3, 256, 256)
print(im.shape)
im = torch.from_numpy(im).to("cuda:0")

lena_original = Image.open("lena.png")

lena = lena_original.resize((256, 256), Image.Resampling.LANCZOS)
print(np.array(lena).shape)

lena = torch.from_numpy(np.array(lena, dtype=np.float32)).to("cuda:0")
lena = lena.reshape(1, 3, 256, 256) / 255.0


path_conf = "models/first_stage_models/vq-f4/config.yaml"
config = OmegaConf.load(path_conf)
model = load_model_from_config(config, "../../model.ckpt")

q, l, info = model.encode(lena)

restored = model.decode(q)

restored = restored.cpu()
restored = restored.detach().numpy()
print(restored)


plt.subplot(1,2,1)
plt.imshow(lena_original)
plt.subplot(1,2,2)
plt.imshow(restored.reshape(256, 256, 3))
plt.show()
