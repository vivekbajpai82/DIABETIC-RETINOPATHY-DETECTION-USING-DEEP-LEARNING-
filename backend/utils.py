import os, requests, io
from PIL import Image

HF_TOKEN = os.getenv("HF_TOKEN")

HF_ENGINE_URL = "https://vivekbajpai82-dr-b5-engine.hf.space/predict_all"

def predict(image: Image.Image):
    try:
        # Image ko bytes mein convert karo
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        
        headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
        
        response = requests.post(
            HF_ENGINE_URL, 
            headers=headers,
            files={"file": ("img.jpg", img_byte_arr.getvalue(), "image/jpeg")},
            timeout=120  # Heatmap banne mein time lagta hai isliye 120 sec
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"HF Space Error {response.status_code}: {response.text}"}

    except Exception as e:
        return {"error": f"Render API Gateway Error: {str(e)}"}