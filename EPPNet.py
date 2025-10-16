# explain_nis.py
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from diffusers import UNet2DModel, 
import safetensors.torch

# Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
output_dir = "./explanations"
os.makedirs(output_dir, exist_ok=True)


with open("ddim-breast_mri-256-segguided-config.json", "r") as f:
    config = json.load(f)

model = UNet2DModel.from_config(config).to(device)
state_dict = safetensors.torch.load_file("ddim-breast_mri-256-segguided.safetensors")
model.load_state_dict(state_dict)
model.eval()
def extract_features(img_tensor):
    feature_maps = []
    def hook_fn(module, input, output):
        feature_maps.append(output.detach())
    
    handle = model.up_blocks[-1].register_forward_hook(hook_fn)
    with torch.no_grad():
        _ = model(img_tensor, torch.tensor(0, device=device))
    handle.remove()
    return feature_maps[-1].cpu().flatten()

real_paths = sorted([os.path.join("real_processed", f) for f in os.listdir("real_processed") if f.endswith(".png")])
fake_paths = sorted([os.path.join("fake_processed", f) for f in os.listdir("fake_processed") if f.endswith(".png")])


print("Extracting features from real images...")
real_features = []
for path in real_paths:
    img = Image.open(path).convert("L")
    img_np = np.array(img) / 255.0
    img_tensor = torch.tensor(img_np).unsqueeze(0).unsqueeze(0).float().to(device)
    mask_tensor = torch.zeros_like(img_tensor)
    x = torch.cat([img_tensor, mask_tensor], dim=1)
    feat = extract_features(x)
    real_features.append(feat)

print(f" Extracted features from {len(real_features)} real images")


for idx, fake_path in enumerate(fake_paths):
    print(f"\nProcessing {fake_path} ({idx+1}/{len(fake_paths)})")
    
    # Load fake image
    img = Image.open(fake_path).convert("L")
    img_np = np.array(img) / 255.0
    img_tensor = torch.tensor(img_np).unsqueeze(0).unsqueeze(0).float().to(device)
    mask_tensor = torch.zeros_like(img_tensor)
    x = torch.cat([img_tensor, mask_tensor], dim=1)

    fake_feat = extract_features(x)
    
    # Compute NIS
    distances = []
    for real_feat in real_features:
        dist = torch.norm(fake_feat - real_feat).item()
        distances.append(-dist)  # negative for similarity
    
    distances = torch.tensor(distances)
    nis_scores = torch.softmax(distances, dim=0).numpy()
    
    
    top_k = 5
    top_indices = np.argsort(nis_scores)[-top_k:][::-1]
    top_scores = nis_scores[top_indices]
    
    # Visualization
    fig, axes = plt.subplots(1, 6, figsize=(15, 3))
    
    
    axes[0].imshow(img_np, cmap='gray')
    axes[0].set_title("Generated Image")
    axes[0].axis('off')
    
    # Top-5 real images
    for i, real_idx in enumerate(top_indices):
        real_img = Image.open(real_paths[real_idx]).convert("L")
        axes[i+1].imshow(np.array(real_img), cmap='gray')
        axes[i+1].set_title(f"Top {i+1}\n(NIS={top_scores[i]:.4f})")
        axes[i+1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"explanation_{idx+1:03d}.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f" Explanation saved for {fake_path}")

print(f"\n All explanations saved to: {os.path.abspath(output_dir)}")
