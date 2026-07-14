from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

import torch
import torch.nn.functional as F

from transformers import (
    BertTokenizer,
    BertForSequenceClassification
)

# ============================================================
# APP
# ============================================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# DEVICE
# ============================================================
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ============================================================
# LOAD MODEL
# ============================================================
MODEL_PATH = "../model"

tokenizer   = BertTokenizer.from_pretrained(MODEL_PATH)
model       = BertForSequenceClassification.from_pretrained(MODEL_PATH)
class_names = torch.load(f"{MODEL_PATH}/class_names.pt")

model.to(device)
model.eval()

# ============================================================
# SCHEMAS
# ============================================================
class TextRequest(BaseModel):
    text: str

class BatchRequest(BaseModel):
    texts: List[str]

# ============================================================
# HOME
# ============================================================
@app.get("/")
def home():
    return {"message": "Dark Pattern Detector API Running"}

# ============================================================
# PREDICT — single (for manual testing)
# ============================================================
@app.post("/predict")
def predict(request: TextRequest):

    inputs = tokenizer(
        request.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=64
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    probs      = F.softmax(logits, dim=1)
    pred_id    = torch.argmax(probs, dim=1).item()
    prediction = class_names[pred_id]
    confidence = probs[0][pred_id].item() * 100

    return {
        "prediction": prediction,
        "confidence": round(confidence, 2)
    }

# ============================================================
# PREDICT BATCH — used by the browser extension
# ============================================================
@app.post("/predict_batch")
def predict_batch(request: BatchRequest):

    if not request.texts:
        return {"predictions": []}

    inputs = tokenizer(
        request.texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=64
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = F.softmax(logits, dim=1)

    predictions = []

    print("\n========== NEW REQUEST ==========")

    for text, prob_row in zip(request.texts, probs):

        idx = prob_row.argmax().item()

        label = class_names[idx]

        confidence = round(prob_row[idx].item(), 4)

        print(f"TEXT: {text}")
        print(f"PREDICTION: {label}")
        print(f"CONFIDENCE: {confidence}")
        print("------------------------")

        predictions.append({
            "label": label,
            "confidence": confidence
        })

    return {"predictions": predictions}