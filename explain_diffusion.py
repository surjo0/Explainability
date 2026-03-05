# explain_diffusion.py
"""
Explainability analysis aligned with Project Proposal:
"Theoretical and Computational Analysis of Convergence and Explainability in Diffusion-Based Generative Models..."

Focus: Visualize internal behavior (noise prediction, score maps) even if output is noisy.
This supports the project's goal: 'develop mathematical tools to interpret the internal behavior'.
"""

import os
import json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from diffusers import UNet2DModel, DDPMScheduler
from PIL import Image

# Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
output_dir = "./results"
os.makedirs(output_dir, exist_ok=True)

# Load config
with open("config.json", "r") as f:
    config = json.load(f)

sample_size = config.get("sample_size", 256)
in_channels = config.get("in_channels", 2)
out_channels = config.get("out_channels", 1)

# Reconstruct model
model = UNet2DModel(
    sample_size=sample_size,
    in_channels=in_channels,
    out_channels=out_channels,
    layers_per_block=config.get("layers_per_block", 2),
    block_out_channels=tuple(config.get("block_out_channels", [128, 256, 512, 512])),
    down_block_types=tuple(config.get("down_block_types", ["DownBlock2D"] * 4)),
    up_block_types=tuple(config.get("up_block_types", ["UpBlock2D"] * 4)),
).to(device)

model.load_state_dict(torch.load("diffusion_pytorch_model.bin", map_location=device))
model.eval()

# Scheduler
scheduler = DDPMScheduler(num_train_timesteps=1000)
num_steps = scheduler.config.num_train_timesteps

# === LOAD REAL DATA ===
img_pil = Image.open("test_lr_mri.png").convert("L")
img_np = np.array(img_pil) / 255.0

mask_pil = Image.open("mask.png").convert("L")
mask_np = np.array(mask_pil).astype(np.float32)

# Tensors
img_tensor = torch.tensor(img_np).unsqueeze(0).unsqueeze(0).float().to(device)
mask_tensor = torch.tensor(mask_np).unsqueeze(0).unsqueeze(0).float().to(device)

# Start from pure noise (for unconditional analysis) OR noisy image (for conditional)
# We use pure noise to test generative capability from scratch
x = torch.cat([torch.randn_like(img_tensor), mask_tensor], dim=1)

print("✅ Input prepared: [noise, multi-class_mask]")

# === LOG INTERNAL BEHAVIOR ===
intermediates = []
noise_pred_magnitudes = []
timesteps_logged = []

print("Sampling with internal diagnostics...")
for step, t in enumerate(reversed(range(num_steps))):
    if t % 50 == 0:  # Changed from 100 to 50
        timesteps_logged.append(t)
        intermediates.append(x.cpu().detach())

    with torch.no_grad():
        noise_pred = model(x, torch.tensor([t], device=device)).sample

    if t % 50 == 0:  # Changed from 100 to 50
        # Noise prediction magnitude (L1 norm per pixel)
        mag = noise_pred.abs().mean(dim=1, keepdim=True)  # [1,1,H,W]
        mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)
        noise_pred_magnitudes.append(mag.cpu())

    x = scheduler.step(noise_pred, t, x).prev_sample

print("✅ Sampling completed.")

# === VISUALIZE: Generation Trajectory ===
cols = 5
rows = int(np.ceil(len(intermediates) / cols))
fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
axes = axes.flatten()

for i, img_t in enumerate(intermediates):
    img = img_t.squeeze(0)[0].numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    axes[i].imshow(img, cmap='gray')
    axes[i].set_title(f"t={timesteps_logged[i]}", fontsize=8)
    axes[i].axis('off')

# Turn off unused subplots
for j in range(i+1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "generation_trajectory.png"), dpi=150, bbox_inches='tight')
plt.close()

# === VISUALIZE: Noise Prediction Magnitude Maps ===
cols = 5
rows = int(np.ceil(len(noise_pred_magnitudes) / cols))
fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
axes = axes.flatten()

for i, mag in enumerate(noise_pred_magnitudes):
    axes[i].imshow(mag[0,0], cmap='hot')
    axes[i].set_title(f"t={timesteps_logged[i]}", fontsize=8)
    axes[i].axis('off')

# Turn off unused subplots
for j in range(i+1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "noise_prediction_maps.png"), dpi=150, bbox_inches='tight')
plt.close()

# === QUANTITATIVE ANALYSIS (Optional) ===
try:
    from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
    import lpips

    # Try to load ground truth
    gt_pil = Image.open("gt_mri.png").convert("L")
    gt_np = np.array(gt_pil) / 255.0
    has_gt = True
except (FileNotFoundError, ImportError) as e:
    print(f"⚠️ Skipping quantitative analysis: {e}")
    has_gt = False

if has_gt:
    # Get final generated image
    final_img = intermediates[-1].squeeze(0)[0].numpy()
    final_img = (final_img - final_img.min()) / (final_img.max() - final_img.min() + 1e-8)
    final_img = final_img.astype(np.float32)
    gt_np = gt_np.astype(np.float32)

    # Compute metrics
    p = psnr(gt_np, final_img, data_range=1.0)
    s = ssim(gt_np, final_img, data_range=1.0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss_fn = lpips.LPIPS(net='alex').to(device)
    img_tensor = torch.from_numpy(final_img).unsqueeze(0).unsqueeze(0).to(device)
    gt_tensor = torch.from_numpy(gt_np).unsqueeze(0).unsqueeze(0).to(device)
    l = loss_fn(img_tensor, gt_tensor).item()

    # Save metrics
    with open(os.path.join(output_dir, "final_metrics.txt"), "w") as f:
        f.write(f"PSNR: {p:.2f}\n")
        f.write(f"SSIM: {s:.4f}\n")
        f.write(f"LPIPS: {l:.4f}\n")

    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(gt_np, cmap='gray')
    axes[0].set_title("Ground Truth")
    axes[0].axis('off')

    axes[1].imshow(final_img, cmap='gray')
    axes[1].set_title(f"Generated (PSNR={p:.2f})")
    axes[1].axis('off')

    diff = np.abs(gt_np - final_img)
    axes[2].imshow(diff, cmap='hot')
    axes[2].set_title("Absolute Difference")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✅ Quantitative analysis saved to: {output_dir}")
else:
    print("No ground truth or missing packages. Skipping metrics.")

# === SAVE RAW DATA FOR FUTURE USE ===
np.save(os.path.join(output_dir, "final_output.npy"), intermediates[-1].squeeze(0)[0].numpy())
np.save(os.path.join(output_dir, "input_mask.npy"), mask_np)
np.save(os.path.join(output_dir, "input_image.npy"), img_np)

print(f"✅ All results saved to: {os.path.abspath(output_dir)}")