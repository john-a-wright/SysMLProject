from compressai.models import CompressionModel
from compressai.models.utils import conv, deconv
import torch.nn as nn
from compressai.entropy_models import EntropyBottleneck
import torch

#zlib

class GDN_1d(nn.Module):
    r"""Generalized Divisive Normalization layer.

    Introduced in `"Density Modeling of Images Using a Generalized Normalization
    Transformation" <https://arxiv.org/abs/1511.06281>`_,
    by Balle Johannes, Valero Laparra, and Eero P. Simoncelli, (2016).

    .. math::

       y[i] = \frac{x[i]}{\sqrt{\beta[i] + \sum_j(\gamma[j, i] * x[j]^2)}}

    """

    def __init__(
        self,
        in_channels: int,
        inverse: bool = False,
        beta_min: float = 1e-6,
        gamma_init: float = 0.1,
    ):
        super().__init__()

        beta_min = float(beta_min)
        gamma_init = float(gamma_init)
        self.inverse = bool(inverse)

        self.beta_reparam = NonNegativeParametrizer(minimum=beta_min)
        beta = torch.ones(in_channels)
        beta = self.beta_reparam.init(beta)
        self.beta = nn.Parameter(beta)

        self.gamma_reparam = NonNegativeParametrizer()
        gamma = gamma_init * torch.eye(in_channels)
        gamma = self.gamma_reparam.init(gamma)
        self.gamma = nn.Parameter(gamma)

    def forward(self, x: Tensor) -> Tensor:
        _, C, _ = x.size()

        beta = self.beta_reparam(self.beta)
        gamma = self.gamma_reparam(self.gamma)
        gamma = gamma.reshape(C, C, 1)
        norm = F.conv1d(x**2, gamma, beta)

        if self.inverse:
            norm = torch.sqrt(norm)
        else:
            norm = torch.rsqrt(norm)

        out = x * norm

        return out

class GDN1_1d(GDN):
    r"""Simplified GDN layer.

    Introduced in `"Computationally Efficient Neural Image Compression"
    <http://arxiv.org/abs/1912.08771>`_, by Johnston Nick, Elad Eban, Ariel
    Gordon, and Johannes Ballé, (2019).

    .. math::

        y[i] = \frac{x[i]}{\beta[i] + \sum_j(\gamma[j, i] * |x[j]|}

    """

    def forward(self, x: Tensor) -> Tensor:
        _, C, _ = x.size()

        beta = self.beta_reparam(self.beta)
        gamma = self.gamma_reparam(self.gamma)
        gamma = gamma.reshape(C, C, 1)
        norm = F.conv1d(torch.abs(x), gamma, beta)

        if not self.inverse:
            norm = 1.0 / norm

        out = x * norm

        return out

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
            nn.GDN1_1d(compressed_d),
            nn.Conv1d(compressed_d, compressed_d, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv1d(compressed_d, compressed_d, kernel_size=3, stride=1, padding=1),
            nn.GDN1_1d(compressed_d),
            nn.ReLU(),
            nn.Conv1d(compressed_d, compressed_d//2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(compressed_d//2, 1, kernel_size=3, stride=1, padding=1),
            nn.GDN1_1d(),
            nn.LazyLinear(compressed_d),
            nn.BatchNorm1d(1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.LazyLinear(compressed_d),
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
            nn.ConvTranspose1d(uncompressed_d//2, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.LazyLinear(uncompressed_d),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.LazyLinear(uncompressed_d)
        )
    
    def forward(self, x):
        y = self.encoder(x)
        
        y_hat, y_likelihoods = self.entropy_bottleneck(y)

        x_hat = self.decoder(y_hat)

        return y_hat, x_hat, y_likelihoods