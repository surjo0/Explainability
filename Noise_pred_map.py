
import os
import json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from diffusers import UNet2DModel, DDIMScheduler
from PIL import Image
import safetensors.torch


device = "cuda" if torch.cuda.is_available() else "cpu"
output_dir = "./results"
os.makedirs(output_dir, exist_ok=True)


with open("ddim-breast_mri-256-segguided-config.json", "r") as f:
    config = json.load(f)

model = UNet2DModel.from_config(config).to(device)
state_dict = safetensors.torch.load_file("ddim-breast_mri-256-segguided.safetensors")
model.load_state_dict(state_dict)
model.eval()



scheduler = DDIMScheduler.from_config("ddim-breast_mri-256-segguided-config.json")
scheduler.set_timesteps(200)  # 200 steps
timesteps = scheduler.timesteps
num_steps = len(timesteps)


img_pil = Image.open("test_lr_mri.png").convert("L")
img_np = np.array(img_pil) / 255.0

mask_pil = Image.open("mask.png").convert("L")
mask_np = np.array(mask_pil).astype(np.float32)


mask_np = (mask_np > 0).astype(np.float32)


img_tensor = torch.tensor(img_np).unsqueeze(0).unsqueeze(0).float().to(device)
mask_tensor = torch.tensor(mask_np).unsqueeze(0).unsqueeze(0).float().to(device)


x = torch.cat([torch.randn_like(img_tensor), mask_tensor], dim=1)
print("✅ Input prepared: [noise, binary_mask]")


intermediates = []
noise_pred_magnitudes = []
timesteps_logged = []

print("Sampling with internal diagnostics...")
for i, t in enumerate(timesteps):
    if i % 50 == 0:  # Log every 50 steps → 0, 50, 100, ..., 150 → 4 logs? Wait—200 steps total
        # But timesteps has 200 elements → i from 0 to 199
        # i % 50 == 0 → i = 0, 50, 100, 150 → 4 steps
        # To get 10 steps, use i % 20 == 0
        # But you asked for "every 50 time steps" — in DDIM 200, that's 4 steps
        # However, you want 10 images → so we'll log every 20 steps (200/10 = 20)
        pass  # We'll fix below


log_indices = np.linspace(0, len(timesteps)-1, 10, dtype=int)
intermediates = []
noise_pred_magnitudes = []
timesteps_logged = []

for i, t in enumerate(timesteps):
    if i in log_indices:
        timesteps_logged.append(t.item())
        intermediates.append(x.cpu().detach())

    with torch.no_grad():
        noise_pred = model(x, t).sample

    if i in log_indices:
        mag = noise_pred.abs().mean(dim=1, keepdim=True)
        mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)
        noise_pred_magnitudes.append(mag.cpu())

    x = scheduler.step(noise_pred, t, x).prev_sample




cols = 5
rows = 2  
fig, axes = plt.subplots(rows, cols, figsize=(15, 6))
axes = axes.flatten()

for i in range(10):  
    img = intermediates[i].squeeze(0)[0].numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    axes[i].imshow(img, cmap='gray')
    axes[i].set_title(f"t={timesteps_logged[i]}", fontsize=14)  # Larger font
    axes[i].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "generation_trajectory.png"), dpi=150, bbox_inches='tight')
plt.close()


fig, axes = plt.subplots(rows, cols, figsize=(15, 6))
axes = axes.flatten()

for i in range(10):
    axes[i].imshow(noise_pred_magnitudes[i][0,0], cmap='hot')
    axes[i].set_title(f"t={timesteps_logged[i]}", fontsize=16)  # Larger font
    axes[i].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "noise_prediction_maps.png"), dpi=150, bbox_inches='tight')
plt.close()


try:
    from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
    import lpips

    gt_pil = Image.open("gt_mri.png").convert("L")
    gt_np = np.array(gt_pil) / 255.0
    has_gt = True
except (FileNotFoundError, ImportError) as e:
    print(f"⚠️ Skipping quantitative analysis: {e}")
    has_gt = False

if has_gt:
    final_img = intermediates[-1].squeeze(0)[0].numpy()
    final_img = (final_img - final_img.min()) / (final_img.max() - final_img.min() + 1e-8)
    final_img = final_img.astype(np.float32)
    gt_np = gt_np.astype(np.float32)

    p = psnr(gt_np, final_img, data_range=1.0)
    s = ssim(gt_np, final_img, data_range=1.0)
    loss_fn = lpips.LPIPS(net='alex').to(device)
    img_tensor = torch.from_numpy(final_img).unsqueeze(0).unsqueeze(0).to(device)
    gt_tensor = torch.from_numpy(gt_np).unsqueeze(0).unsqueeze(0).to(device)
    l = loss_fn(img_tensor, gt_tensor).item()

    with open(os.path.join(output_dir, "final_metrics.txt"), "w") as f:
        f.write(f"PSNR: {p:.2f}\n")
        f.write(f"SSIM: {s:.4f}\n")
        f.write(f"LPIPS: {l:.4f}\n")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(gt_np, cmap='gray')
    axes[0].set_title("Ground Truth")
    axes[0].axis('off')
    axes[1].imshow(final_img, cmap='gray')
    axes[1].set_title(f"Generated (PSNR={p:.2f})")
    axes[1].axis('off')
    axes[2].imshow(np.abs(gt_np - final_img), cmap='hot')
    axes[2].set_title("Absolute Difference")
    axes[2].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f" Quantitative analysis saved to: {output_dir}")

# === SAVE RAW OUTPUTS ===
np.save(os.path.join(output_dir, "final_output.npy"), intermediates[-1].squeeze(0)[0].numpy())
np.save(os.path.join(output_dir, "input_mask.npy"), mask_np)
np.save(os.path.join(output_dir, "input_image.npy"), img_np)

print(f" All results saved to: {os.path.abspath(output_dir)}")