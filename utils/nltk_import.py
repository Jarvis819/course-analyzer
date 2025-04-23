import nltk

print("📦 Downloading NLTK data...")

nltk.download('punkt', force=True)

nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
print("✅ NLTK setup complete.")
