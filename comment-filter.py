from fastapi import FastAPI, Request
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification
import torch

app = FastAPI()

# بارگذاری مدل فارسی
tokenizer = AutoTokenizer.from_pretrained(
    "HooshvareLab/bert-fa-base-uncased-clf-persian-hate-speech"
)
model = AutoModelForSequenceClassification.from_pretrained(
    "HooshvareLab/bert-fa-base-uncased-clf-persian-hate-speech"
)


@app.post("/analyze")
async def analyze_text(request: Request):
    data = await request.json()
    text = data["text"]

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits).item()

    # کلاس‌بندی مدل:
    # 0: normal
    # 1: hate speech

    is_toxic = predicted_class_id == 1
    return {"is_toxic": is_toxic}
