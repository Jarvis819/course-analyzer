# pipeline/summarization.py

from transformers import pipeline

# instantiate once
_summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    # model="sshleifer/distilbart-cnn-12-6",
    
    device=0  # set to -1 if you’re on CPU
)

def generate_summary(liked_reviews, complaint_reviews,
                     max_input_len=1024,
                     max_summary_len=80,
                     min_summary_len=20):
    """
    Returns two concise abstractive summaries (strings):
     - What students liked
     - Common complaints
    """
    def _summarize(text):
        if not text:
            return "No feedback in this category."
        # truncation + single-pass summarization
        return _summarizer(
            text[:max_input_len],
            max_length=max_summary_len,
            min_length=min_summary_len,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True
        )[0]["summary_text"]

    liked_text = " ".join(liked_reviews)
    complaint_text = " ".join(complaint_reviews)

    liked_summary     = _summarize(liked_text)
    complaint_summary = _summarize(complaint_text)

    return liked_summary, complaint_summary

