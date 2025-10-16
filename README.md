# Explainability in Generative Medical Diffusion Models: A Faithfulness-Based Analysis of MRI Synthesis



In this work, we explore explainability for generative diffusion models that synthesize breast MRI images. Unlike many prior studies that rely on multiple classes, our task involves only one anatomical class, focusing purely on image level relationships. We evaluate how different prototype-based explainability methods can connect generated images with the real examples they were influenced by. Our analysis shows that prototype-based reasoning can provide faithful and interpretable explanations even in single class medical domains.

This study suggests that diffusion models can be both accurate and interpretable when their design is closely aligned with the medical task. By linking the generative process with meaningful explanations, we take a step toward more transparent and trustworthy AI for medical imaging.


1. Prerequisites 

    Python 3.8+
    GPU (recommended) or CPU
    ~2 GB disk space

2. Setup

git clone [https://github.com/your-username/explainable-medical-diffusion.git](https://github.com/surjo0/Explainability.git)
cd explainable-medical-diffusion

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install diffusers safetensors scikit-image lpips matplotlib pillow numpy


3. Download Pretrained Model

5. Run Explainability Pipeline

