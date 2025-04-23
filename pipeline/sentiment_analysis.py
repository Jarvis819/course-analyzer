# sentiment_analysis.py

import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from tqdm.auto import tqdm
import os

def load_sentiment_model(model_path):
    print("🔄 Loading sentiment model...")
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    device = 0 if torch.cuda.is_available() else -1
    sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer,truncation=True,padding=True, device=device)
    print("✅ Sentiment model loaded.")
    return sentiment_pipeline

def run_sentiment_analysis(csv_path, output_path, model_path, text_column="Cleaned_Review", batch_size=32):
    df = pd.read_csv(csv_path)
    df[text_column] = df[text_column].fillna("")

    sentiment_pipeline = load_sentiment_model(model_path)

    sentiments = []
    print(f"🔍 Running sentiment analysis on {len(df)} reviews...")

    for i in tqdm(range(0, len(df), batch_size), desc="Sentiment Analysis"):
        batch = df[text_column][i:i+batch_size].tolist()
        results = sentiment_pipeline(batch)
        sentiments.extend([res["label"] for res in results])

    df["Predicted_Sentiment"] = sentiments
    df.to_csv(output_path, index=False)
    print(f"📁 Sentiment results saved to {output_path}")


def analyze_sentiment(reviews, model_path="Jarvis8191/sentiment-model"):
    sentiment_pipeline = load_sentiment_model(model_path)
    results = sentiment_pipeline(reviews)

    labels = [int(r["label"].split("_")[1]) for r in results]  # Convert LABEL_0 -> 0, ..., LABEL_4 -> 4
    avg_rating = sum(labels) / len(labels)
    positivity_percent = 100 * avg_rating / 4  # Max label is 4 (which is 5 stars)

    return labels, positivity_percent

def load_sentiment_pipeline(model_path="Jarvis8191/sentiment-model"):
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=device, truncation=True, padding=True)

def predict_sentiment_for_sentences(sentences, model_path="Jarvis8191/sentiment-model"):
    sentiment_pipeline = load_sentiment_pipeline(model_path)
    results = sentiment_pipeline(sentences)
    # Convert LABEL_0 to 0, LABEL_1 to 1, etc.
    return [int(r["label"].split("_")[1]) for r in results]



if __name__ == "__main__":
    model_checkpoint_path = "Jarvis8191/sentiment-model" # "sentiment_models\checkpoint_epoch_5_v1"  # Adjust if your path differs
    input_csv = "data/sample_reviews.csv"
    output_csv = "data/reviews_with_sentiment.csv"

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    run_sentiment_analysis(input_csv, output_csv, model_checkpoint_path)
