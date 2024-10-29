import torch
import os
from torchvision.utils import save_image
from UNet import UNet
from ddim import beta_scheduler, DDIMSampler
import sys

if __name__ == "__main__":
    num_timesteps = 50
    eta = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    noise_dir = sys.argv[1]
    output_dir = sys.argv[2]
    model_path = sys.argv[3]

    os.makedirs(output_dir, exist_ok=True)

    model = UNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    betas = beta_scheduler(n_timestep=1000)
    ddim_sampler = DDIMSampler(model, betas, num_timesteps, eta)

    noise_images = [torch.load(os.path.join(noise_dir, f'{i:02d}.pt')).to(device) for i in range(10)]
    for i, noise_img in enumerate(noise_images):
        print(f"Processing image {i}")
        generated_img = ddim_sampler.ddim_sample(noise_img)

        generated_img = generated_img.cpu()
        min_val, max_val = torch.min(generated_img), torch.max(generated_img)
        normalized_img = (generated_img - min_val) / (max_val - min_val)

        save_image(normalized_img, os.path.join(output_dir, f'{i:02d}.png'))