import numpy as np
import torch
import torch.nn as nn
import IPython.display as disp
from torch.utils.data import DataLoader
from datasets import load_dataset
from torchvision import transforms
from vector_quantize_pytorch import FSQ
from random import randint

class VectorQuantizedAutoencoder(nn.Module):
    def __init__(self, levels, compressed_d): 
        """
        compressed_d: compressed dimension of the embedding, 128 for SIFT1M
        levels: levels of (Finite )FSQ
        """
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.RELU(),
            nn.Conv1d(32, compressed_d, kernel_size=3, stride=3, padding=1),
            nn.RELU(),
            nn.Conv1d(compressed_d, compressed_d, kernel_size=3, stride=3, padding=1),
            nn.RELU(),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.RELU(),
            nn.Conv1d(compressed_d, compressed_d, kernel_size=6, stride=3, padding=1),
            nn.RELU(),
            nn.AdaptiveAvgPool1d((1,))
        )
        
        self.fsq = FSQ(levels)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 192, kernel_size=6, stride=3, padding=1),
            nn.ConvTranspose2d(192, 192, kernel_size=6, stride=3, padding=0),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.ConvTranspose2d(192, 192, kernel_size=6, stride=3, padding=0),
            nn.Conv2d(192, 3, kernel_size=2, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x, indices = self.fsq(x)
        x = self.decoder(x)

        return x.clamp(-1, 1), indices