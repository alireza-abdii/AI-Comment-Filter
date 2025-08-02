import torch
from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from hazm import Normalizer
import os

# --- Configuration ---
# The path to the fine-tuned model directory
MODEL_DIR = "./phicad_model"

# --- Model and Tokenizer Loading ---
# This section will run only once when the API starts.
print(f"--- Loading model from {MODEL_DIR} ---")

# Check if the model directory exists
if not os.path.exists(MODEL_DIR):
    raise RuntimeError(f"Model directory not found at {MODEL_DIR}. Please make sure you have trained the model and saved it to the correct directory.")

try:
    # Load the fine-tuned model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    normalizer = Normalizer()
    print("✅ Model, tokenizer, and normalizer loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    # If the model fails to load, the application should not start.
    raise

# --- FastAPI Application ---
app = FastAPI()

# Define the request body structure using Pydantic
class CommentRequest(BaseModel):
    text: str

# Define the response body structure
class AnalysisResponse(BaseModel):
    is_toxic: bool
    # You can add more fields here later, e.g., confidence_score

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_text(request: CommentRequest):
    """
    Analyzes a given text to determine if it contains toxic content.
    """
    text_to_analyze = request.text

    # Preprocess the text
    normalized_text = normalizer.normalize(text_to_analyze)
    inputs = tokenizer(normalized_text, return_tensors="pt", truncation=True, padding=True, max_length=128)

    # Make a prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        # The predicted class ID is the one with the highest logit score
        predicted_class_id = torch.argmax(logits, dim=1).item()

    # The notebook maps 'abusive' and 'normal' to labels.
    # We need to know which label corresponds to "toxic".
    # Assuming 'abusive' is the toxic class and its label is 1.
    # This might need to be adjusted based on the actual label_map from your training.
    # You can find the label mapping in the training output or by re-running that cell in the notebook.
    # For now, we'll assume label 1 is toxic.
    is_toxic = predicted_class_id == 1

    return {"is_toxic": is_toxic}

@app.get("/")
def read_root():
    return {"message": "Comment analysis API is running. Send a POST request to /analyze with your text."}
