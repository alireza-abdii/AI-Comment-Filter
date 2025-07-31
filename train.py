import pandas as pd
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, logging
from hazm import Normalizer
import torch
import os

# Set verbosity for transformers to info
logging.set_verbosity_info()

# --- 1. Configuration ---
MODEL_NAME = "HooshvareLab/bert-base-parsbert-uncased"
DATASET_URLS = [
    "https://raw.githubusercontent.com/davardoust/PHICAD/main/PHICAD-part1.csv",
    "https://raw.githubusercontent.com/davardoust/PHICAD/main/PHICAD-part2.csv"
]
OUTPUT_DIR = "./phicad_model"
LOGGING_DIR = './phicad_logs'

def download_and_prepare_dataset():
    """Downloads the PHICAD dataset, merges the parts, and prepares it for training."""
    print("--- Downloading and preparing dataset ---")
    try:
        df1 = pd.read_csv(DATASET_URLS[0], sep="\\t", header=0, on_bad_lines='warn')
        df2 = pd.read_csv(DATASET_URLS[1], sep="\\t", header=0, on_bad_lines='warn')
        df = pd.concat([df1, df2], ignore_index=True)
        print(f"Initial number of rows: {len(df)}")

        # Clean up columns - keep only the ones we need
        expected_columns = ['comment_normalized', 'class']
        df = df[expected_columns]
        
        # Drop rows with missing values in 'class' or 'comment_normalized'
        df.dropna(subset=['class', 'comment_normalized'], inplace=True)
        df = df[df['comment_normalized'].str.strip() != '']

        print(f"Number of rows after cleaning: {len(df)}")

        # Encode labels
        label_encoder = LabelEncoder()
        df['label'] = label_encoder.fit_transform(df['class'])

        # Display class mapping
        label_map = {i: label for i, label in enumerate(label_encoder.classes_)}
        print("Class mapping:", label_map)
        
        # Convert to Hugging Face Dataset object
        dataset = Dataset.from_pandas(df[['comment_normalized', 'label']])
        
        # --- Use a smaller subset for faster training ---
        # Select a small subset of the data
        # Using a very small number for testing purposes.
        if len(dataset) > 2000:
             dataset = dataset.select(range(2000))
        print(f"✅ Dataset prepared and subsetted to {len(dataset)} rows.")
        
        print("✅ Dataset prepared successfully!")
        return dataset, label_map
    except Exception as e:
        print(f"❌ Error downloading or preparing dataset: {e}")
        raise

def main():
    """Main function to run the training pipeline."""
    
    # --- 2. Load and Prepare Data ---
    dataset, label_map = download_and_prepare_dataset()

    # --- 3. Tokenizer and Preprocessing ---
    print("\n--- Initializing tokenizer and normalizer ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    normalizer = Normalizer()

    def preprocess_function(examples):
        # Normalize and tokenize the text
        normalized_texts = [normalizer.normalize(text) for text in examples['comment_normalized']]
        return tokenizer(normalized_texts, truncation=True, padding='max_length', max_length=128)

    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    
    # Split dataset into training and evaluation sets
    splits = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = splits['train']
    eval_dataset = splits['test']
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Evaluation dataset size: {len(eval_dataset)}")

    # --- 4. Model Loading ---
    print("\n--- Loading pre-trained model ---")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=len(label_map),
        id2label={i: label for i, label in label_map.items()},
        label2id={label: i for i, label in label_map.items()}
    )

    # --- 5. Training ---
    print("\n--- Setting up training ---")
    training_args = TrainingArguments(
        output_dir="./phicad_results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        weight_decay=0.01,
        logging_steps=100,
        logging_dir=LOGGING_DIR,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy", # Assuming accuracy is a good metric
    )

    # Define a compute_metrics function
    def compute_metrics(eval_pred):
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary') # Adjust if multi-class
        acc = accuracy_score(labels, predictions)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        # compute_metrics=compute_metrics # Disabled for now to avoid numpy version issues
    )

    print("🚀 Starting training...")
    trainer.train()
    print("✅ Training complete!")

    # --- 6. Evaluation and Saving ---
    print("\n--- Evaluating model ---")
    metrics = trainer.evaluate()
    print("Evaluation metrics:", metrics)

    print(f"\n--- Saving model and tokenizer to {OUTPUT_DIR} ---")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print(f"✅ Model and tokenizer saved successfully to {OUTPUT_DIR}")

if __name__ == "__main__":
    # Add numpy to the imports for compute_metrics if re-enabled
    import numpy as np
    main()
