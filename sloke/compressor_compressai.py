from compressai.models import CompressionModel
from compressai.models.utils import conv, deconv

class Network(CompressionModel):
    def __init__(self, N=128):
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
       y = self.encode(x)
       y_hat, y_likelihoods = self.entropy_bottleneck(y)
       x_hat = self.decode(y_hat)
       return x_hat, y_likelihoods