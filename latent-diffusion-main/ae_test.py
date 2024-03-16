import torch
import json
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
import numpy as np
import matplotlib.pyplot as plt

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


path_conf = "models/first_stage_models/vq-f4/config.yaml"
config = OmegaConf.load(path_conf)
model = load_model_from_config(config, "../model.ckpt")

dummy = torch.ones((1, 3, 256, 256)).to("cuda:0")
print(dummy.shape)

q, l, info = model.encode(dummy)
print(q.shape)
