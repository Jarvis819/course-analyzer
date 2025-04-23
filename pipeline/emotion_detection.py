# emotion_detection.py

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import pandas as pd
from datasets import Dataset
import torch
from tqdm.auto import tqdm
import os

def detect_emotions(input_csv, output_csv, text_column="Cleaned_Review", model_path="emotion_model", batch_size=32):
    # Load model and tokenizer
    # model_name = "monologg/bert-base-cased-goemotions-original"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    # Set device
    device = 0 if torch.cuda.is_available() else -1

    # Load data
    df = pd.read_csv(input_csv)
    reviews = df[text_column].fillna("").tolist()

    # Truncate long reviews
    truncated_reviews = [
        review if len(tokenizer.tokenize(review)) <= 512 else tokenizer.convert_tokens_to_string(tokenizer.tokenize(review)[:512])
        for review in reviews
    ]

    dataset = Dataset.from_dict({"text": truncated_reviews})

    # Emotion detection pipeline
    emotion_pipeline = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=device,
        return_all_scores=True,
        truncation=True,
        padding=True
    )

    # Run detection with progress bar
    predicted_emotions = []
    print("🔍 Running emotion detection...")
    for text in tqdm(dataset["text"], desc="Detecting Emotions"):
        output = emotion_pipeline(text, truncation=True)
        top_emotion = max(output, key=lambda x: x["score"])
        predicted_emotions.append(top_emotion["label"])

    # Save to CSV
    df["Detected_Emotion"] = predicted_emotions
    df.to_csv(output_csv, index=False)
    print(f"✅ Emotion detection complete. Results saved to {output_csv}")


if __name__ == "__main__":
    input_path = "data/Clean_Coursera_reviews_sample.csv"
    output_path = "data/reviews_with_emotions.csv"
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    detect_emotions(input_path, output_path)
