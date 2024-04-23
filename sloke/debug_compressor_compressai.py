from compressor_compressai import Network
from compressai.models import CompressionModel
from compressai.entropy_models import EntropyBottleneck
import numpy as np
import torch.nn as nn
import torch



class Network(CompressionModel):
    def __init__(self, compressed_d = 64, uncompressed_d = 128):
        """
        compressed_d = compressed dimension of the embedding
        """
        super().__init__()
        self.entropy_bottleneck = EntropyBottleneck(1)
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, compressed_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(compressed_d),
            nn.ReLU(),
            nn.Conv1d(compressed_d, compressed_d, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv1d(compressed_d, compressed_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(compressed_d),
            nn.ReLU(),
            nn.Conv1d(compressed_d, compressed_d//2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(compressed_d//2, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.LazyLinear(compressed_d)
        )
        self.encoder_1 = nn.Sequential(
            nn.BatchNorm1d(1)
        )

if __name__ == "__main__":
    model = Network()
    x = torch.randn(64, 1, 128)
    encoded = model.encoder(x)
    print(f"encoded size: {encoded.size()}")
    encoded_1 = model.encoder_1(encoded)
    print(f"encoded_1 size: {encoded_1.size()}")
