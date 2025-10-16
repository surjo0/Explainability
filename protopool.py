
import os
import json
import torch
import numpy as np
from PIL import Image
from diffusers import UNet2DModel
import safetensors.torch
from sklearn.cluster import KMeans

device = "cuda" if torch.cuda.is_available() else "cpu"


with open("ddim-breast_mri-256-segguided-config.json", "r") as f:
    config = json.load(f)

model = UNet2DModel.from_config(config).to(device)
state_dict = safetensors.torch.load_file("ddim-breast_mri-256-segguided.safetensors")
model.load_state_dict(state_dict)
model.eval()


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

real_files = sorted([f for f in os.listdir("./real") if f.endswith(".png")])
fake_files = sorted([f for f in os.listdir("./fake") if f.endswith(".png")])

print("Extracting features...")
real_features = np.array([extract_feature_vector(os.path.join("./real", f)) for f in real_files])
fake_features = np.array([extract_feature_vector(os.path.join("./fake", f)) for f in fake_files[:10]])

M = 100
kmeans = KMeans(n_clusters=M, random_state=42).fit(real_features)
prototype_features = kmeans.cluster_centers_  


protopool_nis = []
for fake_feat in fake_features:
    distances = np.linalg.norm(prototype_features - fake_feat, axis=1)
    similarities = -distances
    exp_sim = np.exp(similarities - np.max(similarities))
    nis = exp_sim / np.sum(exp_sim)
    protopool_nis.append(nis)

protopool_nis = np.array(protopool_nis)
faithfulness_protopool = np.mean(np.max(protopool_nis, axis=1))

print(f" ProtoPool Faithfulness Score: {faithfulness_protopool:.4f}")
np.save("faithfulness_protopool.npy", np.array([faithfulness_protopool]))