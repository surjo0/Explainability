#  Explainability in Generative Medical Diffusion Models  
### A Faithfulness-Based Analysis of MRI Synthesis

This repository contains the implementation and visualization scripts for the paper:  
**“Explainability in Generative Medical Diffusion Models: A Faithfulness-Based Analysis of MRI Synthesis.”**

## 📢 News

**Accepted at the 3rd World Congress on Smart Computing (WCSC 2026).**

This work has been accepted for presentation at **WCSC 2026**. The repository contains the code, evaluation pipeline, and reproducible experiments associated with the paper.

**Preprint available on arXiv:**  
https://arxiv.org/pdf/2602.09781



This paper presents a structured explainability framework for diffusion-based medical image synthesis, with a focus on understanding how generative decisions relate to clinically meaningful image structures. We integrate prototype-based reasoning networks with diffusion models to trace synthetic outputs back to representative training samples, enabling transparent and interpretable generative behavior.

Beyond visual inspection, the work introduces quantitative faithfulness analysis to evaluate how reliably explanation mechanisms reflect the true generative process. We systematically assess structural fidelity, prototype influence, and reconstruction consistency using PSNR, SSIM, LPIPS, and normalized influence-based explainability metrics.


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

```


## 📄 Cite

If you use this repository in your work, please cite:

```bibtex
@article{dey2026explainability,
  title={Explainability in Generative Medical Diffusion Models: A Faithfulness-Based Analysis on MRI Synthesis},
  author={Dey, Surjo and Saikia, Pallabi},
  journal={arXiv preprint arXiv:2602.09781},
  year={2026}
}
