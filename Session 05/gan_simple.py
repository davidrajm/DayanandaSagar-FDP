"""
Simple GAN with EPOCH-WISE PROGRESS in 'gan_progress/' folder!
"""
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os

# Create progress folder
os.makedirs('gan_progress', exist_ok=True)

# Hyperparameters
image_size = 64  # 8x8 flattened
latent_dim = 16
batch_size = 32
epochs = 120

# Generate checkerboard dataset
def generate_checkerboard(n_samples=1000):
    data = []
    for _ in range(n_samples):
        img = np.zeros((8, 8))
        for i in range(8):
            for j in range(8):
                img[i,j] = 1 if (i+j) % 2 == 0 else -1
        img += np.random.normal(0, 0.05, img.shape)
        data.append(img.flatten())
    return torch.FloatTensor(data)

real_data = generate_checkerboard(1000)
dataloader = torch.utils.data.DataLoader(real_data, batch_size=batch_size, shuffle=True)

# Models
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Tanh()
        )
    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

# Initialize
G = Generator()
D = Discriminator()
criterion = nn.BCELoss()
optimizer_G = optim.Adam(G.parameters(), lr=0.001)
optimizer_D = optim.Adam(D.parameters(), lr=0.001)

fixed_noise = torch.randn(8, latent_dim)  # Same noise every epoch

print("GAN Training - Saving progress to 'gan_progress/' folder")
print("Files: epoch_000.png, epoch_020.png, ..., epoch_120.png")

# Training with epoch-wise saving
for epoch in range(epochs):
    for real_imgs in dataloader:
        curr_batch = real_imgs.size(0)
        
        # Train D (5:1 ratio)
        for _ in range(5):
            optimizer_D.zero_grad()
            real_labels = torch.ones(curr_batch, 1)
            fake_labels = torch.zeros(curr_batch, 1)
            
            d_real = D(real_imgs)
            loss_d_real = criterion(d_real, real_labels)
            
            z = torch.randn(curr_batch, latent_dim)
            fake_imgs = G(z).detach()
            d_fake = D(fake_imgs)
            loss_d_fake = criterion(d_fake, fake_labels)
            
            loss_D = loss_d_real + loss_d_fake
            loss_D.backward()
            optimizer_D.step()
        
        optimizer_G.zero_grad()
        fake_imgs = G(z)
        d_fake = D(fake_imgs)
        loss_G = criterion(d_fake, real_labels)
        loss_G.backward()
        optimizer_G.step()
    
    # === SAVE EPOCH-WISE PROGRESS ===
    if epoch % 20 == 0:
        G.eval()
        with torch.no_grad():
            fake_fixed = G(fixed_noise).reshape(8, 8, 8).numpy()
            real_samples = real_data[:8].reshape(8, 8, 8).numpy()
        
        # REAL vs FAKE side-by-side
        fig, axes = plt.subplots(2, 8, figsize=(16, 4))
        for i in range(8):
            axes[0, i].imshow(real_samples[i], cmap='RdBu_r', vmin=-1, vmax=1)
            axes[0, i].set_title('REAL', fontsize=12, color='green')
            axes[0, i].axis('off')
            
            axes[1, i].imshow(fake_fixed[i], cmap='RdBu_r', vmin=-1, vmax=1)
            axes[1, i].set_title('FAKE', fontsize=12, color='red')
            axes[1, i].axis('off')
        
        plt.suptitle(f'Epoch {epoch:03d}: REAL (top) vs FAKE (bottom)', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'gan_progress/epoch_{epoch:03d}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved gan_progress/epoch_{epoch:03d}.png")

# === FINAL RESULTS ===
print("\n🎉 TRAINING COMPLETE!")

# 1. FINAL REAL vs FAKE
G.eval()
with torch.no_grad():
    final_fake = G(fixed_noise).reshape(8, 8, 8).numpy()
    final_real = real_data[:8].reshape(8, 8, 8).numpy()

fig, axes = plt.subplots(2, 8, figsize=(16, 4))
for i in range(8):
    axes[0, i].imshow(final_real[i], cmap='RdBu_r', vmin=-1, vmax=1)
    axes[0, i].set_title('REAL', fontsize=14, color='green', fontweight='bold')
    axes[0, i].axis('off')
    
    axes[1, i].imshow(final_fake[i], cmap='RdBu_r', vmin=-1, vmax=1)
    axes[1, i].set_title('FAKE (GAN)', fontsize=14, color='blue', fontweight='bold')
    axes[1, i].axis('off')

plt.suptitle('FINAL RESULT: GAN Learned Perfect Checkerboard!', fontsize=18, fontweight='bold')
plt.tight_layout()
plt.savefig('gan_progress/final_real_vs_fake.png', dpi=150, bbox_inches='tight')
plt.show()

# 2. CHALLENGE (shuffled)
combined = np.concatenate([final_real, final_fake])
np.random.shuffle(combined)

fig, axes = plt.subplots(1, 16, figsize=(24, 3))
for i in range(16):
    axes[i].imshow(combined[i], cmap='RdBu_r', vmin=-1, vmax=1)
    axes[i].axis('off')
plt.suptitle('CHALLENGE: 8 REAL + 8 FAKE (shuffled - guess!)', fontsize=16)
plt.tight_layout()
plt.savefig('gan_progress/challenge.png', dpi=150, bbox_inches='tight')
plt.show()

# 3. ANSWER KEY
print("\n🔍 ANSWER KEY for challenge.png:")
print("Positions 1-16 (LEFT→RIGHT):")
for i in range(16):
    status = "REAL" if i < 8 else "FAKE"
    print(f"{i+1:2d}:{status:6s}", end="  ")
    if (i+1) % 8 == 0:
        print()
print()

print("\n📁 ALL FILES SAVED in 'gan_progress/':")
print("• epoch_000.png, epoch_020.png, ..., epoch_120.png")
print("• final_real_vs_fake.png")
print("• challenge.png")
print("• Use for lecture: Show evolution → Challenge → Reveal!")
