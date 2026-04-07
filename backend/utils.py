import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import os
import gc
import requests # Hugging Face API call karne ke liye

# ==========================================
# CONFIGURATION
# ==========================================
# Tumhare Hugging Face Space ka API endpoint
HF_API_URL = "https://vivekbajpai82-dr-b5-engine.hf.space/predict_heavy"

# Phase 1 local rahega (EfficientNet-B3 is light enough for Render)
# B3 needs 300x300 input
transform_p1 = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# compatibility for main.py (naming match)
transform_phase1 = transform_p1
transform_phase23 = transform_p1 # Dummy for main.py if needed

# ==========================================
# PHASE-1 LOADER (Local on Render)
# ==========================================
def load_phase1():
    gc.collect()
    model = models.efficientnet_b3(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    
    path = "models/phase_1.pth"
    if os.path.exists(path):
        state_dict = torch.load(path, map_location="cpu")
        # Removing "module." or "model." prefixes if present
        new_state_dict = {k.replace("module.", "").replace("model.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=False)
        del state_dict
    
    model.eval()
    return model

# ==========================================
# MAIN PREDICTION LOGIC
# ==========================================
@torch.no_grad()
def predict(image: Image.Image):
    """
    Step 1: Check Phase 1 locally on Render.
    Step 2: If DR detected, call Hugging Face for Phase 2 & 3.
    """
    image = image.convert("RGB")
    
    try:
        # --- EXECUTE PHASE 1 (LOCAL) ---
        model = load_phase1()
        img_tensor = transform_p1(image).unsqueeze(0)
        output = model(img_tensor)
        
        p1_pred = torch.argmax(output, 1).item()
        p1_conf = torch.softmax(output, dim=1)[0][p1_pred].item()
        
        # Immediate cleanup to save Render RAM
        del model, output
        gc.collect()

        # If Class 0 (No_DR), return immediately
        if p1_pred == 0:
            return {
                "prediction": "No_DR", 
                "confidence": round(p1_conf * 100, 2),
                "engine": "Local-Phase1"
            }

        # --- EXECUTE PHASE 2 & 3 (REMOTE ON HUGGING FACE) ---
        # DR detected! Now we need the heavy B5 models on HF.
        
        # 1. Convert PIL Image to Bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()

        # 2. Send POST request to Hugging Face API
        # Render RAM is safe here as it's just a network call
        response = requests.post(
            HF_API_URL, 
            files={"file": ("image.jpg", img_byte_arr, "image/jpeg")},
            timeout=30 # 30 seconds timeout for safety
        )
        
        if response.status_code == 200:
            hf_result = response.json()
            hf_result["engine"] = "HF-B5-Engine" # Traceability
            return hf_result
        else:
            return {"error": f"Hugging Face Engine Error: {response.status_code}"}

    except Exception as e:
        gc.collect()
        return {"error": f"Backend Error: {str(e)}"}