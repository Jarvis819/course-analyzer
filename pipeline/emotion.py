# emotion_detection.py 
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import pandas as pd
from datasets import Dataset
import torch
from tqdm.auto import tqdm
import os

def detect_emotions_from_list(reviews):
    model_name = "monologg/bert-base-cased-goemotions-original"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    device = 0 if torch.cuda.is_available() else -1

    truncated_reviews = [
        review if len(tokenizer.tokenize(review)) <= 512 else tokenizer.convert_tokens_to_string(tokenizer.tokenize(review)[:512])
        for review in reviews
    ]

    emotion_pipeline = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=device,
        return_all_scores=True,
        truncation=True,
        padding=True
    )

    predicted_emotions = []
    for review in tqdm(truncated_reviews, desc="Detecting Emotions"):
        output = emotion_pipeline(review)
        top_emotion = max(output[0], key=lambda x: x["score"])
        predicted_emotions.append(top_emotion["label"])

    return predicted_emotions
