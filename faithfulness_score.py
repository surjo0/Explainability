# compute_faithfulness_score.py
import os
import json
import torch
import numpy as np
from PIL import Image
from diffusers import UNet2DModel, DDIMScheduler
import safetensors.torch

device = "cuda" if torch.cuda.is_available() else "cpu"

with open("ddim-breast_mri-256-segguided-config.json", "r") as f:
    config = json.load(f)

model = UNet2DModel.from_config(config).to(device)
state_dict = safetensors.torch.load_file("ddim-breast_mri-256-segguided.safetensors")
model.load_state_dict(state_dict)
model.eval()

scheduler = DDIMScheduler.from_config("ddim-breast_mri-256-segguided-config.json")
scheduler.set_timesteps(200)


real_dir = "./real"
real_files = sorted([f for f in os.listdir(real_dir) if f.endswith(".png")])
print(f"Found {len(real_files)} real images.")

def extract_feature_vector(img_path):
    img_pil = Image.open(img_path).convert("L")
    img_np = np.array(img_pil) / 255.0
    img_tensor = torch.tensor(img_np).unsqueeze(0).unsqueeze(0).float().to(device)
    mask_tensor = torch.zeros_like(img_tensor)
    x = torch.cat([img_tensor, mask_tensor], dim=1)

    feature_maps = []
    def hook_fn(module, input, output):
        feature_maps.append(output.detach())
    handle = model.up_blocks[-1].register_forward_hook(hook_fn)

    with torch.no_grad():
        _ = model(x, 0).sample

    pooled = torch.nn.AdaptiveAvgPool2d((1, 1))(feature_maps[-1])
    feature_vector = pooled.flatten().cpu().numpy()
    handle.remove()
    return feature_vector

real_features = []
for fname in real_files:
    feat = extract_feature_vector(os.path.join(real_dir, fname))
    real_features.append(feat)
real_features = np.array(real_features)
print(f"Real features shape: {real_features.shape}")

fake_features = []
num_fake = min(10, len(real_files))

for i in range(num_fake):
    print(f"Generating fake image {i+1}/{num_fake}...")
    
    
    img_pil = Image.open(os.path.join(real_dir, real_files[i])).convert("L")
    img_np = np.array(img_pil) / 255.0
    clean_img = torch.tensor(img_np).unsqueeze(0).unsqueeze(0).float().to(device)
    
    # Add noise at t=500
    t_start = 500
    alpha = scheduler.alphas_cumprod[t_start].to(device)
    noise = torch.randn_like(clean_img)
    noisy_img = torch.sqrt(alpha) * clean_img + torch.sqrt(1 - alpha) * noise

    
    mask_tensor = torch.zeros_like(noisy_img)
    x = torch.cat([noisy_img, mask_tensor], dim=1)


    for t in scheduler.timesteps:
        with torch.no_grad():
            noise_pred = model(x, t).sample
        x = scheduler.step(noise_pred, t, x).prev_sample

   
    generated_image = x[:, 0:1, :, :]  # [1, 1, 256, 256]
    mask_for_feature = torch.zeros_like(generated_image)
    x_gen = torch.cat([generated_image, mask_for_feature], dim=1)  # [1, 2, 256, 256]

    
    feature_maps = []
    def hook_fn(module, input, output):
        feature_maps.append(output.detach())
    handle = model.up_blocks[-1].register_forward_hook(hook_fn)

    with torch.no_grad():
        _ = model(x_gen, 0).sample

    pooled = torch.nn.AdaptiveAvgPool2d((1, 1))(feature_maps[-1])
    fake_feat = pooled.flatten().cpu().numpy()
    fake_features.append(fake_feat)
    handle.remove()

fake_features = np.array(fake_features)
print(f"Fake features shape: {fake_features.shape}")


nis_scores = []
for fake_feat in fake_features:
    distances = np.linalg.norm(real_features - fake_feat, axis=1)
    similarities = -distances
    exp_sim = np.exp(similarities - np.max(similarities))
    nis = exp_sim / np.sum(exp_sim)
    nis_scores.append(nis)

nis_scores = np.array(nis_scores)
faithfulness_score = np.mean(np.max(nis_scores, axis=1))

print(f"\n Faithfulness Score: {faithfulness_score:.4f}")
np.save("faithfulness_score.npy", np.array([faithfulness_score]))