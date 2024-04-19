import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloader
from models import Generator, Discriminator
import os
from torchvision.utils import save_image


def train(data_dir, batch_size, img_size, latent_dim, num_epochs, lr):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = get_dataloader(data_dir, batch_size, img_size)

    generator = Generator(latent_dim).to(device)
    discriminator = Discriminator().to(device)

    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    os.makedirs('generated_images', exist_ok=True)  # 画像保存用ディレクトリを作成

    fixed_noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)  # 固定のノイズを定義


    for epoch in range(num_epochs):
        for i, real_images in enumerate(dataloader):
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            # 識別器の学習
            optimizer_D.zero_grad()
            label_real = torch.ones(batch_size, device=device)
            label_fake = torch.zeros(batch_size, device=device)

            output_real = discriminator(real_images)
            loss_D_real = criterion(output_real, label_real)
            loss_D_real.backward()

            noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
            fake_images = generator(noise)
            output_fake = discriminator(fake_images.detach())
            loss_D_fake = criterion(output_fake, label_fake)
            loss_D_fake.backward()

            loss_D = loss_D_real + loss_D_fake
            optimizer_D.step()

            # 生成器の学習
            optimizer_G.zero_grad()
            output_fake = discriminator(fake_images)
            loss_G = criterion(output_fake, label_real)
            loss_G.backward()
            optimizer_G.step()

            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], "
                      f"Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}")
                # 生成された画像を保存
                with torch.no_grad():
                    fake_images = generator(fixed_noise).detach().cpu()
                    save_image(fake_images, f"generated_images/epoch_{epoch+1}_step_{i+1}.png", normalize=True)

    torch.save(generator.state_dict(), 'generator.pth')
    torch.save(discriminator.state_dict(), 'discriminator.pth')
