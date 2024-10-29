import os
import json
import torch
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms as T, datasets, transforms
from torch.optim import Adam
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import CLIPTokenizer, CLIPTextModel
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from omegaconf import OmegaConf
import gc
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import LambdaLR
import sys
import random
import numpy as np
from ldm.models.diffusion.ddpm import LatentDiffusion
from transformers import AutoTokenizer, AutoModel

# configs
max_grad_norm = 1.0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
learning_rate = 5e-5
min_lr = 1e-6
epochs = 30
accumulation_steps = 1
num_warmup_steps = 0

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed()

def load_word_embeddings(word, embedding_model_name):
    tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
    model = AutoModel.from_pretrained(embedding_model_name)
    word_embedding = model(**tokenizer(word, return_tensors="pt"))["pooler_output"].squeeze(0)
    return word_embedding

def get_warmup_scheduler(optimizer, num_warmup_steps):
    def lr_lambda(current_step):
        return float(current_step) / float(max(1, num_warmup_steps)) if current_step < num_warmup_steps else 1.0
    return LambdaLR(optimizer, lr_lambda)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# Load model
config = OmegaConf.load('./stable-diffusion/configs/stable-diffusion/v1-inference.yaml')
model_params = config.model.params
model = LatentDiffusion(**model_params).to(device).float()
tokenizer = model.cond_stage_model.tokenizer

# Load checkpoint
checkpoint = torch.load('./stable-diffusion/ldm/models/stable-diffusion-v1/model.ckpt', map_location=device)
model.load_state_dict({k: v for k, v in checkpoint["state_dict"].items() if not k.startswith('model_ema.')}, strict=False)
del checkpoint
torch.cuda.empty_cache()
gc.collect()

# Move model to device first
model = model.to(device)

# Keep entire model in fp32
model = model.float()
vgg = models.vgg16(pretrained=True).features[:16].to(device).eval()
for param in vgg.parameters():
    param.requires_grad = False
# Clear memory
torch.cuda.empty_cache()
gc.collect()

def initialize_special_tokens(model, tokens_data):
    tokenizer = model.cond_stage_model.tokenizer
    transformer = model.cond_stage_model.transformer
    token_embeds = transformer.get_input_embeddings()
    for token_name, token_info in tokens_data.items():
        num_added_tokens = tokenizer.add_tokens(token_name)
        transformer.resize_token_embeddings(len(tokenizer))
        token_idx = tokenizer.convert_tokens_to_ids(token_name)
        token_info["token_idx"] = token_idx
        with torch.no_grad():
            if "dog" in token_name.lower():
                similar_tokens = ["puppy", "canine", "dog", "animal"]
            elif "david revoy" in token_name.lower():
                similar_tokens = ["artist", "painter", "illustrator", "David Revoy"]
            else:
                similar_tokens = []
            if similar_tokens:
                similar_ids = [tokenizer.encode(t)[1] for t in similar_tokens if t in tokenizer.vocab]
                if similar_ids:
                    similar_embeddings = token_embeds.weight[similar_ids].clone()
                    token_embeds.weight.data[token_idx] = similar_embeddings.mean(0)
                else:
                    random_indices = torch.randint(0, len(tokenizer) - num_added_tokens, (100,))
                    random_embeddings = token_embeds.weight[random_indices].clone()
                    token_embeds.weight.data[token_idx] = random_embeddings.mean(0)

# Load data
with open('./hw2_data/textual_inversion/input.json', 'r') as f:
    data = json.load(f)
initialize_special_tokens(model, data)

import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, root_folder, tokenizer, is_training=True):
        self.image_paths = []
        self.prompts = []
        self.token_map = {}  # Maps source index to token names for easy association
        self.is_training = is_training
        self.tokenizer = tokenizer

        # Image transformations
        self.transform = transforms.Compose([
            transforms.Resize(512),
            transforms.RandomCrop(512),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        self.eval_transform = transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        # Templates to maintain context during training
        self.templates = {
            "<new1>": [
                "a photo of {}", "a detailed photo of {}", "a picture of {}",
                "a clear photo of {}", "{} in the picture", 
                "high quality photo of {}", "a photo showing {}"
            ],
            "<new2>": [
                "artwork by {}", "illustration by {}", "digital art by {}",
                "painting in the style of {}", "{}'s artistic style",
                "characteristic artwork of {}", "digital painting by {}"
            ]
        }

        # Load images and assign prompts based on token association
        for source_idx, source_data in data.items():
            image_folder = os.path.join(root_folder, source_idx)
            token_name = source_data["token_name"]
            self.token_map[source_idx] = token_name  # Link folder to token

            for fname in os.listdir(image_folder):
                if fname.endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(image_folder, fname))
                    if is_training:
                        template = random.choice(self.templates[token_name])
                        prompt = template.format(token_name)
                    else:
                        prompt = self.templates[token_name][0].format(token_name)  # Consistent prompt for evaluation
                    self.prompts.append(prompt)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image) if self.is_training else self.eval_transform(image)

        prompt = self.prompts[idx]
        return image, prompt

    def __len__(self):
        return len(self.image_paths)

def perceptual_loss(generated, target, vgg):
    generated_features = vgg(generated)
    target_features = vgg(target)
    return F.mse_loss(generated_features, target_features)

def contrastive_loss(positive, negative, margin=1.0):
    positive_distance = F.pairwise_distance(positive, negative)
    return torch.mean(torch.clamp(margin - positive_distance, min=0.0))

def local_feature_loss(generated, target, vgg):
    generated_local_features = vgg(generated)  # Fine-tune as needed
    target_local_features = vgg(target)
    return F.mse_loss(generated_local_features, target_local_features)
# Setup data loader
train_dataset = CustomDataset(data, './hw2_data/textual_inversion/', tokenizer, is_training=True)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1, pin_memory=True, persistent_workers=False)

embedding_layer = model.cond_stage_model.transformer.get_input_embeddings()
special_token_params = [embedding_layer.weight[tokenizer.convert_tokens_to_ids(name)] for name in data.keys()]
optimizer = AdamW([{'params': special_token_params}], lr=learning_rate, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)
scaler = torch.cuda.amp.GradScaler()
warmup_scheduler = get_warmup_scheduler(optimizer, num_warmup_steps)
cosine_scheduler = CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader) // accumulation_steps, eta_min=min_lr)

total_steps = len(train_loader) * epochs // accumulation_steps

current_step = 0
best_loss = float('inf')

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    batch_count = 0
    optimizer.zero_grad()
    for batch_idx, (images, prompts) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")):
        torch.cuda.empty_cache()
        images = images.to(device, non_blocking=True)
        text_embeddings = model.get_learned_conditioning(prompts)
        for token_info in data.values():
            token_idx = token_info["token_idx"]
        with torch.no_grad():
            latents = model.encode_first_stage(images)
            latents = model.get_first_stage_encoding(latents)
        t = torch.randint(model.num_timesteps // 4, model.num_timesteps, (latents.shape[0],), device=device).long()
        noise = torch.randn_like(latents)
        noisy_latents = model.q_sample(x_start=latents, t=t, noise=noise)
        noise_pred = model.apply_model(noisy_latents, t, text_embeddings)
        mse_loss = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()
        generated_image = model.decode_first_stage(latents)
        images_resized = T.Resize((224, 224))(images)
        generated_resized = T.Resize((224, 224))(generated_image)
        p_loss = perceptual_loss(generated_resized, images_resized, vgg)
        decay_loss = 0.001 * torch.norm(embedding_layer.weight[token_idx], p=2) ** 2
        local_loss = local_feature_loss(generated_resized, images_resized, vgg)
        if len(text_embeddings) > 1:
            c_loss = contrastive_loss(text_embeddings[0], text_embeddings[1])
        else:
            c_loss = torch.tensor(0.0, device=device)
        total_loss = mse_loss + 0.3 * p_loss + 0.05 * c_loss + 0.1 * local_loss + decay_loss
        total_loss.backward()  
        torch.cuda.empty_cache()
        if (batch_idx + 1) % accumulation_steps == 0:
            clip_grad_norm_(embedding_layer.parameters(), max_grad_norm)
            optimizer.step()  
            if current_step < num_warmup_steps:
                warmup_scheduler.step()
            else:
                cosine_scheduler.step()
            current_step += 1
            optimizer.zero_grad()  
            torch.cuda.empty_cache()
        epoch_loss += total_loss.item() * accumulation_steps
        batch_count += 1
        del images
        torch.cuda.empty_cache()
        gc.collect()
    avg_loss = epoch_loss / batch_count
    print(f"Epoch {epoch + 1} Loss: {avg_loss:.6f}")
    if avg_loss < best_loss:
        best_loss = avg_loss
        embeddings = []
        for token_name in data.keys():
            token_id = tokenizer.convert_tokens_to_ids(token_name)
            embeddings.append(embedding_layer.weight[token_id].detach().cpu())
        torch.save(embeddings, 'best_embeddings.pth')


torch.cuda.empty_cache()
gc.collect()

# Define output directory
output_dir = './output_textual_inversion/'
model.eval()  # Set model to evaluation mode
# Function to create directories as per required structure
def create_output_directories(output_dir, data):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for source_idx, source_data in data.items():
        source_dir = os.path.join(output_dir, source_idx)
        if not os.path.exists(source_dir):
            os.makedirs(source_dir)
        for prompt_idx, _ in enumerate(source_data['prompt']):
            prompt_dir = os.path.join(source_dir, str(prompt_idx))
            if not os.path.exists(prompt_dir):
                os.makedirs(prompt_dir)

# Function to save generated images
def save_generated_images(images, source_idx, prompt_idx, output_dir):
    for i, image in enumerate(images):
        image_path = os.path.join(output_dir, source_idx, str(prompt_idx), f"source{source_idx}_prompt{prompt_idx}_{i}.png")
        image.save(image_path)

# Create the required directory structure
create_output_directories(output_dir, data)

# Image generation and saving loop
for source_idx, source_data in data.items():
    for prompt_idx, prompt in enumerate(source_data['prompt']):
        generated_images = []

        # Generate 25 images per prompt
        for _ in range(25):
            with torch.no_grad():
                # Obtain text embeddings
                text_embeddings = model.get_learned_conditioning([prompt]).to(device)

                # Define the shape of the latent variables
                batch_size = text_embeddings.size(0)
                image_size = 512  # Assume input image size is 512x512
                downsample_factor = 8  # Latent space is 1/8 of the input image size in each dimension
                latent_shape = (batch_size, model.first_stage_model.embed_dim, image_size // downsample_factor, image_size // downsample_factor)

                # Use the latent diffusion model to sample from noise using p_sample_loop
                latents = model.p_sample_loop(cond=text_embeddings, shape=latent_shape)

                # Decode latents into images
                generated_image = model.decode_first_stage(latents)

                # Process the image to convert to proper format
                generated_image = (generated_image.squeeze(0).permute(1, 2, 0) + 1) / 2  # (C, H, W) -> (H, W, C)
                generated_image = (generated_image * 255).clamp(0, 255).byte().cpu().numpy()

                # Convert to PIL image
                generated_images.append(Image.fromarray(generated_image))

        # Save generated images
        save_generated_images(generated_images, source_idx, prompt_idx, output_dir)