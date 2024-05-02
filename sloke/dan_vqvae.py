import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from vector_quantize_pytorch import FSQ
from random import randint
from torchviz import make_dot

class VectorQuantizedAutoencoder(nn.Module):
    def __init__(self, levels): 
        super().__init__()
        
        self.analysis_transform = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=2, stride=1, padding=1),
            nn.Conv2d(192, 192, kernel_size=6, stride=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.GELU(),
            nn.Conv2d(192, 192, kernel_size=6, stride=3, padding=1),
            nn.Conv2d(192, 512, kernel_size=6, stride=3, padding=1),
        )
        
        self.fsq = FSQ(levels)
        
        self.synthesis_transform = nn.Sequential(
            nn.ConvTranspose2d(512, 192, kernel_size=6, stride=3, padding=1),
            nn.ConvTranspose2d(192, 192, kernel_size=6, stride=3, padding=0),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.ConvTranspose2d(192, 192, kernel_size=6, stride=3, padding=0),
            nn.Conv2d(192, 3, kernel_size=2, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.analysis_transform(x)
        x, indices = self.fsq(x)
        x = self.synthesis_transform(x)

        return x.clamp(-1, 1), indices

if __name__ == "__main__":
    x = torch.rand(1, 1, 128)
    levels = [8, 8, 8, 5, 5, 5]

    model = VectorQuantizedAutoencoder(levels)

    dot = make_dot(model(x), params = dict(model.named_parameters()))
    dot.render("vqvae_fsq", format="png")
