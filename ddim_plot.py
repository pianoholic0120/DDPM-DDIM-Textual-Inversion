import torch
import numpy as np
from ddim import DDIMSampler, beta_scheduler
from torchvision.utils import save_image
import os
from UNet import UNet

# 定義球面線性插值 (slerp)
def slerp(val, low, high):
    omega = torch.acos(torch.clamp(torch.dot(low/torch.norm(low), high/torch.norm(high)), -1, 1))
    so = torch.sin(omega)
    return torch.sin((1.0 - val) * omega) / so * low + torch.sin(val * omega) / so * high

# 定義線性插值
def lerp(val, low, high):
    return (1.0 - val) * low + val * high

if __name__ == "__main__":
    torch.cuda.empty_cache()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 加載預訓練的模型和DDIM
    model = UNet()
    model.load_state_dict(torch.load('./hw2_data/face/UNet.pt'))
    model.to(device)
    model.eval()

    betas = beta_scheduler(n_timestep=1000)
    ddim_sampler = DDIMSampler(model, betas, num_sampling_steps=50, eta=0)

    # 加載噪聲圖片 00.pt 和 01.pt
    noise_00 = torch.load('hw2_data/face/noise/00.pt').to(device)
    noise_01 = torch.load('hw2_data/face/noise/01.pt').to(device)

    # 插值參數 α
    alphas = np.linspace(0.0, 1.0, 11)

    output_dir_slerp = 'output_folder_ddim/slerp/'
    output_dir_lerp = 'output_folder_ddim/lerp/'
    os.makedirs(output_dir_slerp, exist_ok=True)
    os.makedirs(output_dir_lerp, exist_ok=True)

    for i, alpha in enumerate(alphas):
        print(f"Processing interpolation {i} with alpha={alpha}")

        # 進行 slerp 插值
        interp_noise_slerp = slerp(alpha, noise_00.view(-1), noise_01.view(-1)).view_as(noise_00)
        # 確保形狀正確 [batch_size, channels, height, width]
        interp_noise_slerp = interp_noise_slerp.unsqueeze(0)  # 添加 batch 維度
        if interp_noise_slerp.dim() == 5:
            interp_noise_slerp = interp_noise_slerp.squeeze(1)  # 移除不必要的維度

        # generated_img_slerp = ddim_sampler.ddim_sample(interp_noise_slerp).cpu().clamp(-1, 1)
        generated_img_slerp = ddim_sampler.ddim_sample(interp_noise_slerp).cpu()
        min_val = torch.min(generated_img_slerp).cpu()
        max_val = torch.max(generated_img_slerp).cpu()
        normalized_img_slerp = (generated_img_slerp - min_val) / (max_val - min_val).cpu()
        save_image(normalized_img_slerp, os.path.join(output_dir_slerp, f'slerp_{i:02d}.png'))

        # 進行線性插值
        interp_noise_lerp = lerp(alpha, noise_00.view(-1), noise_01.view(-1)).view_as(noise_00)
        # 確保形狀正確 [batch_size, channels, height, width]
        interp_noise_lerp = interp_noise_lerp.unsqueeze(0)  # 添加 batch 維度
        if interp_noise_lerp.dim() == 5:
            interp_noise_lerp = interp_noise_lerp.squeeze(1)  # 移除不必要的維度

        # generated_img_lerp = ddim_sampler.ddim_sample(interp_noise_lerp).cpu().clamp(-1, 1)
        generated_img_lerp = ddim_sampler.ddim_sample(interp_noise_lerp).cpu()
        min_val = torch.min(generated_img_lerp).cpu()
        max_val = torch.max(generated_img_lerp).cpu()
        normalized_img_lerp = (generated_img_lerp - min_val) / (max_val - min_val).cpu()
        save_image(normalized_img_lerp, os.path.join(output_dir_lerp, f'lerp_{i:02d}.png'))


    print("All interpolations done!")
