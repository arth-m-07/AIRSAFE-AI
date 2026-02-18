from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import torch
import numpy as np
import cv2
import segmentation_models_pytorch as smp
from io import BytesIO
from PIL import Image

app = FastAPI()

# ✅ CORS FIX (IMPORTANT)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = smp.Unet(
    encoder_name="resnet18",
    encoder_weights=None,
    in_channels=3,
    classes=10,
)

model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()

@app.post("/predict/")
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")
    image_np = np.array(image)

    img = image_np / 255.0
    img_tensor = torch.tensor(img).permute(2,0,1).unsqueeze(0).float().to(device)

    with torch.no_grad():
        output = model(img_tensor)
        pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    # ---------- SAFE LANDING ANALYSIS ----------
    total_pixels = pred_mask.size
    safe_pixels = np.sum(pred_mask == 0)   # Assuming class 0 = sand
    safe_ratio = safe_pixels / total_pixels

    if safe_ratio > 0.4:
        status = "SAFE LANDING ZONE DETECTED"
    else:
        status = "HIGH RISK - NO SAFE ZONE"

    safety_score = round(safe_ratio * 100, 2)

    # ---------- Visualization ----------
    np.random.seed(42)
    colors = np.random.randint(0,255,(10,3))
    color_mask = colors[pred_mask]

    overlay = cv2.addWeighted(image_np, 0.6, color_mask.astype(np.uint8), 0.4, 0)

    # Add status text on image
    cv2.putText(
        overlay,
        status,
        (20,50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0) if safe_ratio > 0.4 else (0,0,255),
        2,
        cv2.LINE_AA
    )

    _, buffer = cv2.imencode(".png", overlay)

    return {
        "status": status,
        "safety_score": safety_score,
        "image": buffer.tobytes().hex()
    }

