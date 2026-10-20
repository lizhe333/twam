"""Torch port of the installed baseline augmax geometry and HSV jitter.

Equations follow augmax (Copyright 2024 Konrad Heidler, Apache-2.0).
Preserve its default hue=.1, jitter probability=.5, and the installed version's
no-op saturation assignment. Random streams differ between JAX and Torch.
"""
import math
import torch
import torch.nn.functional as F


def augment(images, parameters=None):
    b, _, h, w = images.shape
    if (h, w) != (224, 224):
        raise ValueError("Augmentation expects the baseline's 224x224 images")
    if parameters is None:
        u = torch.rand((b, 7), device=images.device)
        parameters = torch.stack([(u[:, 0]-.5)*12, (u[:, 1]-.5)*12,
                                  (u[:, 2]-.5)*10*math.pi/180,
                                  (u[:, 3]-.5)*.6, (u[:, 4]-.5)*.8,
                                  (u[:, 5]-.5)*.2, (u[:, 6]<.5).float()], 1)
    dy, dx, angle, brightness, contrast, hue_delta, apply = [p[:, None, None] for p in parameters.unbind(1)]
    y, x = torch.meshgrid(torch.arange(h, device=images.device)-111.5,
                          torch.arange(w, device=images.device)-111.5, indexing="ij")
    scale = 212/224
    gy = scale*(angle.cos()*y + angle.sin()*x) + dy
    gx = scale*(-angle.sin()*y + angle.cos()*x) + dx
    grid = torch.stack((gx/111.5, gy/111.5), -1)
    images = F.grid_sample(images.float(), grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    value, arg = images.max(1)
    span = value-images.min(1).values
    second = images.gather(1, ((arg+1)%3)[:, None]).squeeze(1)
    third = images.gather(1, ((arg+2)%3)[:, None]).squeeze(1)
    hue = torch.where(span==0, 0., (2*arg+(second-third)/span.clamp_min(1e-20))/6)
    saturation = torch.where(value==0, 0., span/value.clamp_min(1e-20))
    val = torch.where(brightness<0, value*(1+brightness), value*(1-brightness)+brightness)
    slant = torch.tan((contrast+1)*math.pi/4)
    # Algebraically equivalent to augmax, stable when contrast is exactly zero.
    p1 = slant/(2*(1+slant))
    val = torch.where(val<p1, val/slant,
                      torch.where(val>1-p1, val/slant+1-1/slant, slant*(val-.5)+.5))
    k = (images.new_tensor([5, 3, 1])[None, :, None, None] + (hue+hue_delta)[:, None]*6)%6
    colors = val[:, None]-val[:, None]*saturation[:, None]*torch.minimum(k, 4-k).clamp(0, 1)
    return torch.where(apply[:, None].bool(), colors, images)
