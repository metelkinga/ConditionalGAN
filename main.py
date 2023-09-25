import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.utils as vutils
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Generator
class Generator(nn.Module):
    def __init__(self, noise_dim, num_classes, image_channels):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, noise_dim)
        self.model = nn.Sequential(
            nn.Linear(noise_dim + noise_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, image_channels * 256 * 256),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        noise = torch.cat((noise, self.label_emb(labels)), dim=1)
        img = self.model(noise)
        img = img.view(img.size(0), image_channels, 256, 256)
        return img


# Discriminator
class Discriminator(nn.Module):
    def __init__(self, image_channels, num_classes):
        super(Discriminator, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, 256)
        self.model = nn.Sequential(
            nn.Linear(image_channels * 256 * 256 + 256, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, image, labels):
        img_flat = image.view(image.size(0), -1)
        label_embedding = self.label_embedding(labels)
        x = torch.cat((img_flat, label_embedding), dim=1)
        validity = self.model(x)
        return validity


# Hyperparameters
batch_size = 1
num_epochs = 100
lr = 0.0001
noise_dim = 10
num_classes = 2  # Healthy and Tumorous
image_channels = 1  # Grayscale MRI images


# Image transformation
class GrayscaleToTensor(object):
    def __call__(self, pic):
        return torch.from_numpy(np.array(pic, np.float32, copy=False) / 255.0).unsqueeze(0)


# Data preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.Grayscale(num_output_channels=1),
    GrayscaleToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load the dataset
train_dataset = ImageFolder(root='data/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Load the validation dataset
val_dataset = ImageFolder(root='data/val', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize models and optimizers
generator = Generator(noise_dim, num_classes, image_channels).to(device)
discriminator = Discriminator(image_channels, num_classes).to(device)
optimizer_G = optim.AdamW(generator.parameters(), lr=lr)
optimizer_D = optim.AdamW(discriminator.parameters(), lr=lr)
criterion = nn.BCELoss()

if not os.path.exists('result'):
    os.makedirs('result')

validation_frequency = 1

# Training loop
for epoch in range(num_epochs):
    d_loss_total = 0.0
    g_loss_total = 0.0

    for batch_idx, (images, labels) in enumerate(train_loader):
        batch_size = images.shape[0]
        real_labels = torch.ones(batch_size, 1, device=device)
        fake_labels = torch.zeros(batch_size, 1, device=device)

        # Training Discriminator
        optimizer_D.zero_grad()
        real_outputs = discriminator(images, labels)
        real_loss = criterion(real_outputs, real_labels)

        noise = torch.randn(batch_size, noise_dim, device=device)
        fake_images = generator(noise, labels)
        fake_outputs = discriminator(fake_images.detach(), labels)
        fake_loss = criterion(fake_outputs, fake_labels)

        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer_D.step()

        # Training Generator
        optimizer_G.zero_grad()
        noise = torch.randn(batch_size, noise_dim, device=device)
        fake_images = generator(noise, labels)
        fake_outputs = discriminator(fake_images, labels)
        g_loss = criterion(fake_outputs, real_labels)
        g_loss.backward()
        optimizer_G.step()

        d_loss_total += d_loss.item()
        g_loss_total += g_loss.item()

    # Calculate average losses for the epoch
    avg_d_loss = d_loss_total / len(train_loader)
    avg_g_loss = g_loss_total / len(train_loader)

    # Validation
    if epoch % validation_frequency == 0:
        generator.eval()
        with torch.no_grad():
            val_d_loss = 0.0
            val_g_loss = 0.0
            for batch_idx, (val_images, val_labels) in enumerate(val_loader):
                val_images = val_images.to(device)
                val_labels = val_labels.to(device)
                batch_size_val = val_images.size(0)

                val_real_outputs = discriminator(val_images, val_labels)
                val_real_labels = torch.ones(batch_size_val, 1, device=device)
                val_real_loss = criterion(val_real_outputs, val_real_labels)

                val_fake_labels = torch.randint(0, num_classes, (batch_size_val,), device=device)
                val_fake_noise = torch.randn(batch_size_val, noise_dim, device=device)
                val_fake_images = generator(val_fake_noise, val_fake_labels)
                val_fake_outputs = discriminator(val_fake_images, val_fake_labels)
                val_fake_labels = torch.zeros(batch_size_val, 1, device=device)
                val_fake_loss = criterion(val_fake_outputs, val_fake_labels)

                val_d_loss += val_real_loss.item() + val_fake_loss.item()
                val_g_loss += val_fake_loss.item()

            avg_val_d_loss = val_d_loss / len(val_loader)
            avg_val_g_loss = val_g_loss / len(val_loader)

        generator.train()

        print(f"Epoch [{epoch}/{num_epochs}] D_loss: {avg_d_loss:.4f} G_loss: {avg_g_loss:.4f}")
        print(f"Val_D_loss: {avg_val_d_loss:.4f} Val_G_loss: {avg_val_g_loss:.4f}")

    # Generating images
    if epoch % 10 == 0:
        with torch.no_grad():
            fake_labels = torch.randint(0, num_classes, (1,), device=device)
            fake_sample = generator(torch.randn(1, noise_dim, device=device), fake_labels)
        image_path = os.path.join('result', f"wygenerowany_obraz_epoka_{epoch}.png")
        vutils.save_image(fake_sample, image_path, normalize=True)

torch.save(generator.state_dict(), 'generator.pth')
