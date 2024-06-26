import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from vector_quantize_pytorch import FSQ
from random import randint
from torchviz import make_dot
from torchsummary import summary

class VectorQuantizedAutoencoder(nn.Module):
    def __init__(self, levels, compressed_d, uncompressed_d = 128): 
        """
        uncompressed_d: uncompressed dimension of the embedding, 128 for SIFT1M
        compressed_d: compressed dimension of the embedding
        levels: levels of (Finite )FSQ
        """
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, compressed_d, kernel_size=3, stride=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(compressed_d, compressed_d, kernel_size=3, stride=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv1d(compressed_d, compressed_d, kernel_size=3, stride=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d((1,))
        )
        
        self.fsq = FSQ(levels)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(1, 32, kernel_size=3, stride=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, compressed_d, kernel_size=3, stride=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(compressed_d, uncompressed_d, kernel_size=3, stride=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(uncompressed_d, uncompressed_d, kernel_size=3, stride=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d((1,))
        )

    def forward(self, x):
        bottleneck = self.encoder(x)
        bottleneck = torch.permute(bottleneck, (0, 2, 1))

        bottleneck, indices = self.fsq(bottleneck)

        reconstructed = self.decoder(bottleneck)
        reconstructed = torch.permute(reconstructed, (0, 2, 1))

        return reconstructed.clamp(-1, 1), indices

if __name__ == "__main__":
    # test for dimensionality
    levels =  [8, 8, 8, 5, 5, 5]
    model = VectorQuantizedAutoencoder(levels, len(levels)).to(
        torch.device("cuda")
    )

    x = torch.rand(1, 1, 128)
    # vis_graph = make_dot(model(x), params = dict(model.named_parameters()))
    # vis_graph.render("vqvae_fsq", format="png")
    print(summary(model, (1,128)))

    # embedding = torch.randn(1, 1, 128)
    # encoded = model.encoder(embedding)
    # encoded = torch.permute(encoded, (0, 2, 1))
    # encoded, indices = model.fsq(encoded)
    # print(f"Shape of encoded vector: {encoded.shape}")
    # print(f"Index of the fsq code-book: {indices}")

    # decoded = model.decoder(encoded)
    # decoded = torch.permute(decoded, (0, 2, 1))
    # print(f"Shape of decoded vector: {decoded.shape}")