import torch.nn as nn

# Define the Generator
class Generator(nn.Module):
    def __init__(self, latent_dim, output_channels=1, img_size=256):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, output_channels * img_size * img_size),  # Adjust the output size
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), 1, self.img_size, self.img_size)  # Adjust the dimensions
        return img

# Define the Discriminator
class Discriminator(nn.Module):
    def __init__(self, input_channels=1, img_size=256):
        super(Discriminator, self).__init__()
        self.img_size = img_size

        self.model = nn.Sequential(
            nn.Linear(input_channels * img_size * img_size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity




