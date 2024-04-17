from compressai.models import CompressionModel
from compressai.models.utils import conv, deconv
import torch.nn as nn
from compressai.entropy_models import EntropyBottleneck
import torch


class Network(CompressionModel):
    def __init__(self, compressed_d = 64, uncompressed_d = 128):
        """
        compressed_d = compressed dimension of the embedding
        """
        super().__init__()
        self.entropy_bottleneck = EntropyBottleneck(compressed_d)
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
            nn.AdaptiveAvgPool1d((1,)),
            nn.Linear(compressed_d, compressed_d),
            nn.ReLU(),
            nn.Linear(compressed_d, compressed_d)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(1, 32, kernel_size=3, stride=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, compressed_d, kernel_size=3, stride=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(compressed_d, uncompressed_d, kernel_size=3, stride=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(uncompressed_d, uncompressed_d, kernel_size=3, stride=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d((1,)),
            nn.Linear(uncompressed_d, uncompressed_d),
            nn.ReLU(),
            nn.Linear(uncompressed_d, uncompressed_d)
        )
    
    def forward(self, x):
        y = self.encoder(x)
        y_hat, y_likelihoods = self.entropy_bottleneck(y)

        y_hat = torch.permute(y_hat, (0, 2, 1))
        x_hat = self.decoder(y_hat)
        x_hat = torch.permute(x_hat, (0, 2, 1))

        return x_hat, y_likelihoods