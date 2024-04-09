class VectorQuantizedAutoencoder(nn.Module):
    def __init__(self, levels): 
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=2, stride=1, padding=1),
            nn.Conv2d(192, 192, kernel_size=6, stride=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.GELU(),
            nn.Conv2d(192, 192, kernel_size=6, stride=3, padding=1),
            nn.Conv2d(192, 512, kernel_size=6, stride=3, padding=1),
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