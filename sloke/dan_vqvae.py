import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from vector_quantize_pytorch import FSQ
from random import randint

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
    img = torch.rand(1, 3, 256, 256)
    levels = [8,5,5,5]

    model = VectorQuantizedAutoencoder(levels)
    encode = model.analysis_transform(img)
    print("shape of the synthesis_transformed data: ${encode.shape}")

    encode, indices = model.fsq(encode)
