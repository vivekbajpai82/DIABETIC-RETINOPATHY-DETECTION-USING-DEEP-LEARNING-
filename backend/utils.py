import torch, os, gc, requests, io
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from huggingface_hub import hf_hub_download

# --- CONFIG ---
# Repo ID ekdum sahi hona chahiye
REPO_ID = "vivekbajpai82/dr-models" 
HF_ENGINE_URL = "https://vivekbajpai82-dr-b5-engine.hf.space/predict_heavy"
HF_TOKEN = os.getenv("HF_TOKEN") # Ye abhi bhi Phase 2/3 ke liye chahiye hoga

# Phase 1 Transform
transform_p1 = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_phase1():
    try:
        # Repo Public hai, toh bina token ke bhi download hoga!
        model_path = hf_hub_download(
            repo_id=REPO_ID,
            filename="phase_1.pth",
            cache_dir="models"
        )
        
        model = models.efficientnet_b3(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
        
        state_dict = torch.load(model_path, map_location="cpu")
        new_state_dict = {k.replace("module.", "").replace("model.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=False)
        model.eval()
        return model
    except Exception as e:
        print(f"Error downloading/loading Phase 1: {e}")
        return None

@torch.no_grad()
def predict(image: Image.Image):
    image = image.convert("RGB")
    try:
        # --- PHASE 1 (LOCAL) ---
        p1_model = load_phase1()
        if p1_model is None:
            return {"error": "Phase 1 model not loaded"}

        img_tensor = transform_p1(image).unsqueeze(0)
        out = p1_model(img_tensor)
        p1_pred = torch.argmax(out, 1).item()
        p1_conf = torch.softmax(out, 1)[0][p1_pred].item()
        
        del p1_model; gc.collect()

        # Agar No DR hai, toh result turant bhej do
        if p1_pred == 0:
            return {"prediction": "No_DR", "confidence": round(p1_conf * 100, 2)}

        # --- PHASE 2 & 3 (REMOTE CALL TO HF) ---
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        
        # Space ke liye headers (Agar space private hai toh zaroori hai)
        headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
        
        response = requests.post(
            HF_ENGINE_URL, 
            headers=headers,
            files={"file": ("img.jpg", img_byte_arr.getvalue(), "image/jpeg")},
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"HF Space Error: {response.status_code}"}

    except Exception as e:
        return {"error": f"Backend Error: {str(e)}"}