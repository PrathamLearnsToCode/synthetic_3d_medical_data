import torch
from dataloader import CardiacDataset
import imgaug.augmenters as iaa
from pathlib import Path
import torch.nn as nn
from model import Generator, Discriminator
import os
import numpy as np

# Set up augmentation sequence
seq = iaa.Sequential([
    iaa.Affine(scale=(0.85, 1.15), rotate=(-45, 45)),
    iaa.ElasticTransformation()
])

# Set up paths
path = Path("Preprocessed/train/")
output_dir = "/Users/pratham/Desktop/synthetic data"
input_dir = os.path.join(output_dir, "inputs")
mask_dir = os.path.join(output_dir, "masks")

# Create directories if they don't exist
os.makedirs(input_dir, exist_ok=True)
os.makedirs(mask_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Initialize the dataset and dataloader
dataset = CardiacDataset(path, seq)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True, pin_memory=False)

# Initialize the generator and discriminator
latent_dim = 100
output_channels = 1
img_size = 256
generator = Generator(latent_dim, output_channels, img_size)
discriminator = Discriminator(input_channels=output_channels, img_size=img_size)

# Set up loss function and optimizers
adversarial_loss = nn.BCELoss()
generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.001, betas=(0.5, 0.999))
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001, betas=(0.5, 0.999))

# Training loop
num_epochs = 500
for epoch in range(num_epochs):
    for i, (real_images, _) in enumerate(dataloader):
        discriminator_optimizer.zero_grad()

        real_labels = torch.ones(real_images.size(0), 1)
        fake_labels = torch.zeros(real_images.size(0), 1)

        real_images, real_labels = real_images.to(device), real_labels.to(device)

        real_outputs = discriminator(real_images)
        real_loss = adversarial_loss(real_outputs, real_labels)

        z = torch.randn(real_images.size(0), latent_dim).to(device)
        fake_images = generator(z)
        fake_outputs = discriminator(fake_images.detach())
        fake_loss = adversarial_loss(fake_outputs, fake_labels)

        total_discriminator_loss = real_loss + fake_loss
        total_discriminator_loss.backward()
        discriminator_optimizer.step()

        generator_optimizer.zero_grad()

        fake_outputs = discriminator(fake_images)
        generator_loss = adversarial_loss(fake_outputs, real_labels)

        generator_loss.backward()
        generator_optimizer.step()

        # Save synthetic data
        if i % 100 == 0:
            synthetic_input_path = os.path.join(input_dir, f"synthetic_input_epoch_{epoch}_batch_{i}.npy")
            synthetic_mask_path = os.path.join(mask_dir, f"synthetic_mask_epoch_{epoch}_batch_{i}.npy")

            synthetic_input_np = fake_images.detach().cpu().numpy()
            synthetic_mask_np = fake_outputs.detach().cpu().numpy()

            np.save(synthetic_input_path, synthetic_input_np)
            np.save(synthetic_mask_path, synthetic_mask_np)

            print(f"Synthetic data saved: {synthetic_input_path}, {synthetic_mask_path}")