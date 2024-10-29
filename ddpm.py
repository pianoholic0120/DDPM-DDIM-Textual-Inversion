import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset, ConcatDataset, WeightedRandomSampler
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm

# Constants
TIMESTEPS = 1000
IMAGE_CHANNELS = 3
IMAGE_SIZE = 28
NUM_CLASSES = 20  # 10 digits * 2 datasets
MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]

# Paths
OUTPUT_FOLDER = './Output_folder/'
MNISTM_FOLDER = os.path.join(OUTPUT_FOLDER, 'mnistm')
SVHN_FOLDER = os.path.join(OUTPUT_FOLDER, 'svhn')
CLASSIFIER_MODEL_PATH = './Classifier.pth'

# Ensure output folders exist
os.makedirs(MNISTM_FOLDER, exist_ok=True)
os.makedirs(SVHN_FOLDER, exist_ok=True)

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_residual=True):
        super().__init__()
        self.use_residual = use_residual
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
        )
        if use_residual:
            self.residual = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        h = self.conv(x)
        return h + self.residual(x) if self.use_residual else h

class ContextUnet(nn.Module):
    def __init__(self, in_channels, n_feat=256, n_classes=20):
        super().__init__()
        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_classes = n_classes

        self.init_conv = ConvBlock(in_channels, n_feat)
        self.down1 = nn.Sequential(ConvBlock(n_feat, n_feat), nn.MaxPool2d(2))
        self.down2 = nn.Sequential(ConvBlock(n_feat, 2 * n_feat), nn.MaxPool2d(2))
        self.to_vec = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.GELU())

        # Modify time embedding layers
        self.timeembed1 = nn.Sequential(
            nn.Linear(1, n_feat),
            nn.GELU(),
            nn.Linear(n_feat, 2*n_feat)
        )
        self.timeembed2 = nn.Sequential(
            nn.Linear(1, n_feat),
            nn.GELU(),
            nn.Linear(n_feat, n_feat)
        )       
        self.contextembed1 = nn.Sequential(nn.Linear(n_classes, 2*n_feat), nn.GELU(), nn.Linear(2*n_feat, 2*n_feat))
        self.contextembed2 = nn.Sequential(nn.Linear(n_classes, n_feat), nn.GELU(), nn.Linear(n_feat, n_feat))
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 7, 7),
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )

        self.up1 = nn.ModuleList([
            nn.ConvTranspose2d(4 * n_feat, n_feat, 2, 2),
            ConvBlock(n_feat, n_feat),
            ConvBlock(n_feat, n_feat)
        ])
        self.up2 = nn.ModuleList([
            nn.ConvTranspose2d(2 * n_feat, n_feat, 2, 2),
            ConvBlock(n_feat, n_feat),
            ConvBlock(n_feat, n_feat)
        ])
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )

    def forward(self, x, c, t, context_mask):
        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        hiddenvec = self.to_vec(down2)

        c = F.one_hot(c, num_classes=self.n_classes).float() * (1 - context_mask[:, None])

        # Modify how we handle the time embedding
        t = t.unsqueeze(1)  # Add an extra dimension to make it [B, 1]
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

        up1 = self.up0(hiddenvec)
        up2 = self.up1[0](torch.cat([cemb1*up1 + temb1, down2], 1))
        up2 = self.up1[1](up2)
        up2 = self.up1[2](up2)
        up3 = self.up2[0](torch.cat([cemb2*up2 + temb2, down1], 1))
        up3 = self.up2[1](up3)
        up3 = self.up2[2](up3)
        out = self.out(torch.cat([up3, x], 1))
        return out

class DDPM(nn.Module):
    def __init__(self, nn_model, betas, n_T, device, drop_prob=0.1):
        super().__init__()
        self.nn_model = nn_model.to(device)
        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()

        # Register DDPM schedule buffers
        for k, v in self.ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

    def ddpm_schedules(self, beta1, beta2, T):
        assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"
        
        beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
        sqrt_beta_t = torch.sqrt(beta_t)
        alpha_t = 1 - beta_t
        log_alpha_t = torch.log(alpha_t)
        alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()
        
        sqrtab = torch.sqrt(alphabar_t)
        oneover_sqrta = 1 / torch.sqrt(alpha_t)
        
        sqrtmab = torch.sqrt(1 - alphabar_t)
        mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

        return {
            "alpha_t": alpha_t,  # \alpha_t
            "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
            "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
            "alphabar_t": alphabar_t,  # \bar{\alpha_t}
            "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
            "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
            "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
        }

    def forward(self, x, c):
        _ts = torch.randint(1, self.n_T + 1, (x.shape[0],), device=self.device)
        noise = torch.randn_like(x)
        x_t = self.sqrtab[_ts, None, None, None] * x + self.sqrtmab[_ts, None, None, None] * noise
        context_mask = torch.bernoulli(torch.zeros_like(c) + self.drop_prob).to(self.device)
        return self.loss_mse(noise, self.nn_model(x_t, c, _ts / self.n_T, context_mask))

    def sample(self, n_sample, size, device, c, guide_w=0.0):
        x_i = torch.randn(n_sample, *size, device=device)
        c_i = c.to(device)
        context_mask = torch.zeros_like(c_i, device=device)
        c_i = torch.cat([c_i, c_i])
        context_mask = torch.cat([context_mask, torch.ones_like(context_mask, device=device)])

        x_i_store = []
        for i in tqdm(range(self.n_T, 0, -1)):
            t_is = torch.full((n_sample * 2,), i / self.n_T, device=device)
            x_i = x_i.repeat(2, 1, 1, 1)
            z = torch.randn(n_sample, *size, device=device) if i > 1 else 0

            # Batch the eps calculation
            eps = self.nn_model(x_i, c_i, t_is, context_mask)
            eps1, eps2 = eps.chunk(2)
            x_i = x_i[:n_sample]
            eps = (1 + guide_w) * eps1 - guide_w * eps2

            x_i = self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i]) + self.sqrt_beta_t[i] * z
            if i % 20 == 0 or i == self.n_T or i < 8:
                x_i_store.append(x_i.cpu())

        return x_i, x_i_store

class DigitDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, dataset_label=0):
        self.labels = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.dataset_label = dataset_label

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.labels.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        digit_label = self.labels.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, digit_label, self.dataset_label

def load_data(batch_size=32):
    data_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    mnistm_dataset = DigitDataset(
        csv_file='./hw2_data/digits/mnistm/train.csv',
        root_dir='./hw2_data/digits/mnistm/data/',
        transform=data_transforms,
        dataset_label=0
    )

    svhn_dataset = DigitDataset(
        csv_file='./hw2_data/digits/svhn/train.csv',
        root_dir='./hw2_data/digits/svhn/data/',
        transform=data_transforms,
        dataset_label=1
    )

    combined_dataset = ConcatDataset([mnistm_dataset, svhn_dataset])
    
    weights = [1.0/len(mnistm_dataset)] * len(mnistm_dataset) + [1.0/len(svhn_dataset)] * len(svhn_dataset)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    return DataLoader(combined_dataset, batch_size=batch_size, sampler=sampler)

def train_model(model, train_loader, num_epochs=100, lr=1e-4):
    model.train()
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = GradScaler()

    for epoch in range(num_epochs):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            images, digit_labels, dataset_labels = [x.to(device) for x in batch]
            
            combined_labels = digit_labels + dataset_labels * 10
            combined_labels = combined_labels.to(device)

            optimizer.zero_grad()

            with autocast():
                loss = model(images, combined_labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')
        scheduler.step()

        # Save checkpoint every 50 epochs
        if (epoch + 1) % 50 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f'checkpoint_epoch_{epoch+1}.pth')

@torch.no_grad()
def sample_images(model, num_samples=50, guidance_scale=1.5, device='cuda'):
    model.eval()
    model.to(device)
    for dataset_label, dataset_name in enumerate(['mnistm', 'svhn']):
        dataset_folder = os.path.join(OUTPUT_FOLDER, dataset_name)
        os.makedirs(dataset_folder, exist_ok=True)

        for digit in range(10):
            # Create conditional label (digit + dataset)
            c = torch.full((num_samples,), digit + dataset_label * 10, device=device)
            with autocast():
                samples, _ = model.sample(num_samples, (IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE), device, c, guide_w=guidance_scale)

            samples = (samples + 1) / 2  # Denormalize
            samples = torch.clamp(samples, 0, 1)

            for i in range(num_samples):
                img_filename = os.path.join(dataset_folder, f'{digit}_{i+1:03d}.png')
                save_image(samples[i], img_filename)

    print("Image sampling completed.")

def main():
    # Initialize model
    nn_model = ContextUnet(in_channels=IMAGE_CHANNELS, n_feat=256, n_classes=NUM_CLASSES)
    model = DDPM(nn_model, betas=(1e-4, 0.02), n_T=TIMESTEPS, device=device).to(device)

    # Load data
    train_loader = load_data(batch_size=32)

    # Train model
    train_model(model, train_loader, num_epochs=50)

    # Sample images
    sample_images(model, num_samples=50, device=device)

    # Evaluate generated images (assuming digit_classifier.py exists)
    os.system(f'python3 digit_classifier.py --folder {OUTPUT_FOLDER} --checkpoint {CLASSIFIER_MODEL_PATH}')

if __name__ == "__main__":
    main()