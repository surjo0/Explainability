#  Explainability in Generative Medical Diffusion Models  
### A Faithfulness-Based Analysis of MRI Synthesis

This repository contains the implementation and visualization scripts for the paper:  
**“Explainability in Generative Medical Diffusion Models: A Faithfulness-Based Analysis of MRI Synthesis.”**

Our work explores how diffusion-based generative models can produce realistic medical images while remaining interpretable through prototype-based reasoning. Using the DUKE Breast MRI dataset, we evaluate several prototype explainability networks (PPNet, EPPNet, and ProtoPool) to connect generated samples with real data and measure explainability through faithfulness-based metrics.


---

## 📘 Dataset

We use the **Breast Cancer MRI dataset** available from  
[The Cancer Imaging Archive (TCIA)](https://www.cancerimagingarchive.net/collection/duke-breast-cancer-mri/).

This dataset provides dynamic contrast-enhanced (DCE) MRI scans of breast tissue, including both normal and abnormal cases.  
All images are preprocessed with:
- Intensity normalization  
- Bias-field correction  
- Spatial registration  
- Binary mask extraction for breast regions  

Only a single anatomical class (breast MRI) is used for training.

---

## Model Overview

### Diffusion Model
A conditional diffusion model is trained to synthesize high-fidelity MRI images from Gaussian noise.  
The model follows a denoising diffusion implicit model (DDIM) process, gradually reconstructing anatomy through reverse noise estimation steps.

### Prototype-Based Explainability
Three prototype-based methods are used to interpret how generated images relate to training data:
- **PPNet (ProtoPNet):** Basic prototype reasoning using nearest feature patches.  
- **EPPNet (Enhanced ProtoPNet):** Adds normalization and diversity regularization for more stable explanations.  
- **ProtoPool:** Introduces a shared prototype pool for more flexible feature matching.

Each method computes **Normalized Influence Scores (NIS)** to determine how strongly prototypes contribute to generated outputs.

---

## 📊 Quantitative Evaluation

**Image Quality Metrics**
| Metric | Mean ± SD |
|---------|------------|
| PSNR | 19.37 ± 1.67 |
| SSIM | 0.6530 ± 0.1052 |
| LPIPS | 0.2893 ± 0.1050 |

**Explainability Metrics**
| Model | Faithfulness Score |
|--------|--------------------|
| PPNet | 0.0965 |
| EPPNet | **0.1534** |
| ProtoPool | 0.1420 |



---

## ⚙️ How to Run

```bash
# 1️⃣ Clone the repository
git clone https://github.com/surjo0/Explainability.git
cd Explainability

# 2️⃣ Create environment and install dependencies
python -m venv env
source env/bin/activate   # or .\env\Scripts\activate on Windows
pip install -r requirements.txt

# 3️⃣ Run diffusion synthesis and explainability
python duke.py
