import os
import json
import torch
from PIL import Image
from tqdm import tqdm
import sys
from ldm.models.diffusion.ddpm import LatentDiffusion
from omegaconf import OmegaConf
import random
import numpy as np
from torch.cuda.amp import autocast

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed()

def load_model(model_path, config_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = OmegaConf.load(config_path)
    model_params = config.model.params
    model = LatentDiffusion(**model_params).to(device).float()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(
        {k: v for k, v in checkpoint["state_dict"].items() if not k.startswith('model_ema.')},
        strict=False
    )
    model.eval()
    return model  

def inject_special_tokens(model, embeddings_path, data):
    embedding_layer = model.cond_stage_model.transformer.get_input_embeddings()
    loaded_embeddings = torch.load(embeddings_path)
    for i, (source_idx, source_data) in enumerate(data.items()):
        token_name = source_data["token_name"]
        token_id = model.cond_stage_model.tokenizer.convert_tokens_to_ids(token_name)
        embedding_layer.weight.data[token_id] = loaded_embeddings[i].to(embedding_layer.weight.device)
    return model

def generate_images(model, prompt, device, num_images=25, batch_size=5):
    """Generates a specified number of images from a single prompt using batch processing."""
    generated_images = []
    text_embeddings = model.get_learned_conditioning([prompt]).to(device)

    # Generate images in batches
    for i in range(0, num_images, batch_size):
        current_batch_size = min(batch_size, num_images - i)
        with torch.no_grad(), autocast():
            latents = model.p_sample_loop(
                cond=text_embeddings.repeat(current_batch_size, 1, 1),
                shape=(current_batch_size, model.first_stage_model.embed_dim, 64, 64)
            )
            decoded_images = model.decode_first_stage(latents)
            
            for j in range(decoded_images.size(0)):
                img = ((decoded_images[j].permute(1, 2, 0) + 1) / 2 * 255).clamp(0, 255).byte().cpu().numpy()
                generated_images.append(Image.fromarray(img))

    return generated_images

def save_images(images, token_idx, prompt_idx, output_dir):
    """Saves images in the specified output folder structure."""
    prompt_folder = os.path.join(output_dir, str(token_idx), str(prompt_idx))
    os.makedirs(prompt_folder, exist_ok=True)

    for i, img in enumerate(images):
        img.save(os.path.join(prompt_folder, f"source{token_idx}_prompt{prompt_idx}_{i}.png"))

if __name__ == "__main__":
    json_path = sys.argv[1]
    output_folder = sys.argv[2]
    model_path = sys.argv[3]
    config_path = './stable-diffusion/configs/stable-diffusion/v1-inference.yaml'
    
    # Load the main model
    model = load_model(model_path, config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Inject special tokens
    with open(json_path, 'r') as f:
        data = json.load(f)
    embeddings_path = "./best_embeddings.pth"
    model = inject_special_tokens(model, embeddings_path, data)
    
    # Generate images for each token and prompt in the specified structure
    for token_idx, token_data in data.items():
        for prompt_idx, prompt in enumerate(token_data["prompt"]):
            images = generate_images(model, prompt, device, num_images=25)
            save_images(images, token_idx, prompt_idx, output_folder)
