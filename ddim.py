import torch
import torch.nn.functional as F
import numpy as np
from torchvision.utils import save_image
from torchvision import transforms
import os
from PIL import Image
from UNet import UNet
from tqdm import tqdm

class DDIMSampler:
    def __init__(self, model, betas, num_sampling_steps=50, eta=0):
        self.model = model
        self.num_sampling_steps = num_sampling_steps
        self.eta = eta
        self.betas = betas

        # Calculate alpha-related values
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # Set up DDIM-specific timesteps
        step_ratio = len(self.betas) // self.num_sampling_steps
        self.ddim_timesteps = np.arange(1, len(self.betas) + 1, step_ratio)
        self.ddim_timesteps_prev = np.append(np.array([0]), self.ddim_timesteps[:-1])

    def _extract(self, tensor, timesteps, shape):
        batch_size = timesteps.shape[0]
        out = tensor.to(timesteps.device).gather(0, timesteps).float()
        return out.reshape(batch_size, *((1,) * (len(shape) - 1)))

    def _predict_x0(self, x_t, noise, alpha_cumprod_t):
        pred_x0 = (x_t - torch.sqrt(1.0 - alpha_cumprod_t) * noise) / torch.sqrt(alpha_cumprod_t)
        return torch.clamp(pred_x0, min=-1.0, max=1.0)

    def _compute_sigma(self, alpha_cumprod_t, alpha_cumprod_t_prev):
        return self.eta * torch.sqrt(
            (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * 
            (1 - alpha_cumprod_t / alpha_cumprod_t_prev)
        )

    @torch.no_grad()
    def ddim_sample(self, initial_noise):
        device = initial_noise.device
        batch_size = initial_noise.shape[0]
        current_sample = initial_noise

        for i in tqdm(reversed(range(self.num_sampling_steps)), desc="DDIM Sampling"):
            timestep = torch.full((batch_size,), self.ddim_timesteps[i], device=device, dtype=torch.long)
            prev_timestep = torch.full((batch_size,), self.ddim_timesteps_prev[i], device=device, dtype=torch.long)

            # Get alpha_cumprod for current and previous timesteps
            alpha_cumprod_t = self._extract(self.alphas_cumprod, timestep, current_sample.shape)
            alpha_cumprod_t_prev = self._extract(self.alphas_cumprod, prev_timestep, current_sample.shape)

            # Predict noise
            predicted_noise = self.model(current_sample, timestep)

            # Predict x0 and compute sigma
            predicted_x0 = self._predict_x0(current_sample, predicted_noise, alpha_cumprod_t)
            sigma_t = self._compute_sigma(alpha_cumprod_t, alpha_cumprod_t_prev)

            # Compute "direction pointing to x_t"
            pred_dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev - sigma_t**2) * predicted_noise

            # Compute x_{t-1}
            x_prev = torch.sqrt(alpha_cumprod_t_prev) * predicted_x0 + pred_dir_xt + sigma_t * torch.randn_like(current_sample)

            current_sample = x_prev

        return current_sample

def beta_scheduler(n_timestep=1000, linear_start=1e-4, linear_end=2e-2):
    betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
    return betas

if __name__ == "__main__":
    torch.cuda.empty_cache()
    num_timesteps = 50
    eta = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = UNet()
    model.load_state_dict(torch.load('./hw2_data/face/UNet.pt'))
    model.to(device)
    model.eval()

    betas = beta_scheduler(n_timestep=1000)

    ddim_sampler = DDIMSampler(model, betas, num_timesteps, eta)

    noise_dir = 'hw2_data/face/noise/'
    gt_dir = 'hw2_data/face/GT/'
    output_dir = 'output_folder_ddim/'
    os.makedirs(output_dir, exist_ok=True)

    noise_images = [torch.load(os.path.join(noise_dir, f'{i:02d}.pt')).to(device) for i in range(10)]
    gt_images = [transforms.ToTensor()(Image.open(os.path.join(gt_dir, f'{i:02d}.png'))).to(device) for i in range(10)]

    total_mse = 0
    for i, noise_img in enumerate(noise_images):
        print(f"Processing image {i}")
        generated_img = ddim_sampler.ddim_sample(noise_img)

        # generated_img = generated_img.cpu().clamp(0, 1)
        generated_img = generated_img.cpu()
        min_val = torch.min(generated_img).cpu()
        max_val = torch.max(generated_img).cpu()
        normalized_img = (generated_img - min_val) / (max_val - min_val).cpu()


        # print(f'Generated Image Shape: {generated_img.shape}')
        # print(f'Generated Image Type: {generated_img.dtype}')
        # print(f'Generated Image Min: {generated_img.min().item()}, Max: {generated_img.max().item()}')

        # 保存圖像
        save_image(normalized_img, os.path.join(output_dir, f'{i:02d}.png'))

        # 讀取 GT 圖像
        gt_image = gt_images[i].cpu()

        # 計算 MSE
        mse = F.mse_loss(normalized_img, gt_image)

        total_mse += mse.item()
        print(f'Image {i:02d}: MSE = {mse.item()}')

    average_mse = total_mse / len(noise_images)
    print(f'Average MSE: {average_mse}')