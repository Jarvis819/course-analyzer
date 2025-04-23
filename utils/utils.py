import re
import pandas as pd
from transformers import pipeline
import streamlit as st
# ----------------------------
# Text Cleaning Utility
# ----------------------------

def clean_text(text):
    """
    Cleans input text by removing URLs, special characters, and extra whitespaces.
    """
    text = str(text)
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", text)  # Remove special chars
    text = re.sub(r"\s{2,}", " ", text)  # Remove extra whitespace
    return text.strip()


# ----------------------------
# Emotion Categorization
# ----------------------------

liked_emotions = {
    "joy", "gratitude", "admiration", "approval", "excitement",
    "love", "optimism", "relief", "pride", "desire", "amusement"
}

complaint_emotions = {
    "disappointment", "frustration", "confusion", "annoyance",
    "disapproval", "embarrassment", "grief", "nervousness", "fear", "anger"
}

def categorize_emotion(label):
    """
    Categorizes the emotion label into 'liked', 'complaint', or 'neutral'.
    """
    if label.lower() in liked_emotions:
        return "liked"
    elif label.lower() in complaint_emotions:
        return "complaint"
    else:
        return "neutral"


# ----------------------------
# File Reading Utility
# ----------------------------

def read_reviews_from_csv(filepath, column_name="Review"):
    """
    Reads reviews from a CSV and returns a list of cleaned review texts.
    """
    df = pd.read_csv(filepath)
    reviews = df[column_name].fillna("").tolist()
    return [clean_text(review) for review in reviews]


# ----------------------------
# Summarization Model Loader
# ----------------------------


import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns

def get_emotion_distribution_plot(emotions):

    final_emotions = {
        "admiration", "approval", "gratitude", "love", "joy", 
        "disappointment", "confusion", "disapproval", "neutral", "anger"
    }
    filtered_emotions = [e for e in emotions if e in final_emotions]

    theme = st.get_option("theme.base")
    bg_color = "#0e1117" if theme == "dark" else "#ffffff"
    text_color = "#ffffff" if theme == "dark" else "#262730"
    bar_color = "#ff4b4b"  # change this if you want another emotion color

    counts = Counter(filtered_emotions)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=list(counts.keys()), y=list(counts.values()), palette=[bar_color]*len(counts), ax=ax)

    ax.set_xlabel("Emotion", color=text_color)
    ax.set_ylabel("Count", color=text_color)
    ax.set_title("Emotion Distribution", color=text_color)
    ax.tick_params(colors=text_color)
    ax.set_facecolor(bg_color)
    fig.patch.set_facecolor(bg_color)
    sns.despine()
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    ax.tick_params(colors=text_color)
    plt.tight_layout()
    for spine in ax.spines.values():
        spine.set_visible(False)
    return fig


def categorize_emotion_batches(reviews, emotions):
    """
    Splits reviews into 'liked' and 'complaint' categories based on corresponding emotions.

    Parameters:
        reviews (List[str]): List of review texts.
        emotions (List[str]): List of detected emotions for each review.

    Returns:
        Tuple[List[str], List[str]]: (liked_reviews, complaint_reviews)
    """
    liked_reviews = []
    complaint_reviews = []

    for review, emo in zip(reviews, emotions):
        category = categorize_emotion(emo)
        if category == "liked":
            liked_reviews.append(review)
        elif category == "complaint":
            complaint_reviews.append(review)

    return liked_reviews, complaint_reviews

def get_sentiment_progress_plot(positive_percent):
    """
    positive_percent: float (0–100)
    Returns a matplotlib Figure showing a horizontal bar up to that percentage.
    """
    fig, ax = plt.subplots(figsize=(6, 1.2))
    ax.barh([0], [positive_percent], color='#2a9d8f', height=0.6)
    ax.set_xlim(0, 100)
    ax.set_yticks([])  # hide the y axis tick
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Overall Positive Sentiment (%)")
    ax.set_title("", pad=10)

    # Show the percentage on the bar
    ax.text(positive_percent + 1, 0, f"{positive_percent:.1f}%", va='center', fontsize=10)

    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    return fig




import re

import re

def clean_and_capitalize_sentences(text):
    # If it’s a list of strings (e.g. multiple summary chunks), join with a period
    if isinstance(text, list):
        text = ". ".join(text)

    # Now text is definitely one big string:
    text = text.strip()

    # Split on end‑of‑sentence punctuation
    sentences = re.split(r'(?<=[\.\!\?])\s+', text)

    # Capitalize first word of each sentence
    sentences = [s.capitalize() for s in sentences if s]

    # Re‑join into a paragraph
    return " ".join(sentences)


def get_sentiment_distribution_plot(sentiment_labels, bar_color="#0A76F6"):
    """
    Creates a bar plot showing the distribution of predicted star ratings (1 to 5).
    `sentiment_labels` should be a list of integers from 0 to 4 (representing 1 to 5 stars).
    """
    import matplotlib.pyplot as plt
    from collections import Counter

    # Convert model labels (0–4) to star ratings (1–5)
    ratings = [label + 1 for label in sentiment_labels]
    counts = Counter(ratings)

    theme = st.get_option("theme.base")
    bg_color = "#0e1117" if theme == "dark" else "#ffffff"
    text_color = "#ffffff" if theme == "dark" else "#262730"

    fig, ax = plt.subplots()
    ax.bar(counts.keys(), counts.values(), color=bar_color)
    ax.set_xlabel("Predicted Star Rating", color=text_color)
    ax.set_ylabel("Number of Reviews", color=text_color)
    ax.set_xticks(range(1, 6))
    ax.set_title("Predicted Sentiment Distribution", color=text_color)
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    ax.tick_params(colors=text_color)
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    return fig
