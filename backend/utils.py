import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import os
import gc
import requests
from huggingface_hub import hf_hub_download

# --- CONFIG ---
HF_TOKEN = os.getenv("HF_TOKEN")
REPO_ID = "vivekbajpai82/dr-models" 
HF_ENGINE_URL = "https://vivekbajpai82-dr-b5-engine.hf.space/predict_heavy"

# Phase 1 Transform
transform_p1 = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# compatibility aliases for main.py (ERROR FIX)
transform_phase1 = transform_p1
transform_phase23 = transform_p1 

def load_phase1():
    # Phase 1 download logic
    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename="phase_1.pth",
        use_auth_token=HF_TOKEN,
        cache_dir="models"
    )
    
    model = models.efficientnet_b3(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    
    state_dict = torch.load(model_path, map_location="cpu")
    new_state_dict = {k.replace("module.", "").replace("model.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    return model

@torch.no_grad()
def predict(image: Image.Image):
    image = image.convert("RGB")
    try:
        # --- PHASE 1 (Local on Render) ---
        p1_model = load_phase1()
        img_tensor = transform_p1(image).unsqueeze(0)
        out = p1_model(img_tensor)
        p1_pred = torch.argmax(out, 1).item()
        p1_conf = torch.softmax(out, 1)[0][p1_pred].item()
        
        del p1_model; gc.collect()

        if p1_pred == 0:
            return {"prediction": "No_DR", "confidence": round(p1_conf * 100, 2)}

        # --- PHASE 2 & 3 (API call to Hugging Face Space) ---
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()

        response = requests.post(
            HF_ENGINE_URL, 
            files={"file": ("image.jpg", img_byte_arr, "image/jpeg")}
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"HF Engine Error: {response.status_code}"}

    except Exception as e:
        return {"error": str(e)}