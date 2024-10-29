import os
import torch
from torch.cuda.amp import autocast
from torchvision.utils import save_image
from tqdm import tqdm
from ddpm import DDPM, ContextUnet

# Constants
TIMESTEPS = 1000
IMAGE_CHANNELS = 3
IMAGE_SIZE = 28
NUM_CLASSES = 20  # 10 digits * 2 datasets
CHECKPOINT_PATH = './checkpoint_epoch_50.pth'

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(checkpoint_path):
    # Initialize model
    nn_model = ContextUnet(in_channels=IMAGE_CHANNELS, n_feat=256, n_classes=NUM_CLASSES)
    model = DDPM(nn_model, betas=(1e-4, 0.02), n_T=TIMESTEPS, device=device).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded checkpoint from {checkpoint_path}")
    
    return model

@torch.no_grad()
def sample_images(model, output_folder, num_samples=50, guidance_scale=1.5, device='cuda'):
    model.eval()
    model.to(device)
    os.makedirs(output_folder, exist_ok=True)
    
    for dataset_label, dataset_name in enumerate(['mnistm', 'svhn']):
        dataset_folder = os.path.join(output_folder, dataset_name)
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

    print(f"Image sampling completed. Images are saved in {output_folder}")

def main(output_folder):
    # Load model with weights from checkpoint
    model = load_model(CHECKPOINT_PATH)

    # Sample images and save them to the output folder
    sample_images(model, output_folder, num_samples=50, device=device)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python hw2_1_inference.py <output_directory>")
        sys.exit(1)

    output_folder = sys.argv[1]
    main(output_folder)
