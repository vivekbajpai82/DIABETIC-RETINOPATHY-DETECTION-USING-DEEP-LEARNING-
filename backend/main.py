from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import io
from PIL import Image
from utils import predict

app = FastAPI()

# Netlify se request allow karne ke liye (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "Render Backend is Live 🚀"}

# Frontend ki khushi ke liye dummy quality check
@app.post("/check_quality")
async def check_quality(file: UploadFile = File(...)):
    return {"quality_score": 90, "is_good": True}

@app.post("/predict")
async def predict_route(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        result = predict(image)
        return result
    except Exception as e:
        return {"error": str(e)}