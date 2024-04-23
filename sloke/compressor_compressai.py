from compressai.models import CompressionModel
from compressai.models.utils import conv, deconv
import torch.nn as nn
from compressai.entropy_models import EntropyBottleneck
import torch

#zlib


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
            nn.LazyLinear(compressed_d),
            nn.BatchNorm1d(1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.LazyLinear(compressed_d),
            nn.Dropout(0.5),
            nn.BatchNorm1d(1)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, compressed_d, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(compressed_d),
            nn.ConvTranspose1d(compressed_d, uncompressed_d, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(uncompressed_d, uncompressed_d, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(uncompressed_d),
            nn.ConvTranspose1d(uncompressed_d, uncompressed_d, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(uncompressed_d, uncompressed_d//2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(uncompressed_d//2, uncompressed_d//2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(uncompressed_d//2),
            nn.ConvTranspose1d(uncompressed_d//2, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.LazyLinear(uncompressed_d),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(1),
            nn.LazyLinear(uncompressed_d),
            nn.Dropout(0.5),
            nn.BatchNorm1d(1)
        )
    
    def forward(self, x):
        y = self.encoder(x)
        
        y_hat, y_likelihoods = self.entropy_bottleneck(y)

        x_hat = self.decoder(y_hat)

        return y_hat, x_hat, y_likelihoods