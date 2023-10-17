import random

import numpy as np
import torch


class Fourier_noise(object):
    def __init__(self, crop=224, eps=2, size=4, xx=0, yy=0):
        self.eps = eps
        self.size = size
        self.crop = crop
        self.xx = xx
        self.yy = yy

    def __call__(self, x):
        noise_H = self.size * self.xx
        noise_W = self.size * self.yy
        noise_add_data = torch.zeros(3, self.crop, self.crop)
        noise_place = torch.zeros(self.crop, self.crop)
        noise_place[noise_H, noise_W] = 1
        for j in range(3):
            temp_noise = torch.fft.ifftshift(noise_place)
            noise_base = torch.fft.ifft2(temp_noise).real
            noise_base /= torch.linalg.norm(noise_base)
            noise_base *= self.eps
            noise_base *= random.randrange(-1, 2, 2)
            noise_add_data[j] = torch.clamp(x[j] + noise_base, min=0.0, max=1.0)

        return noise_add_data
