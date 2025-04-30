# 📊 AI-Powered E-Learning Review Analyzer

This web app analyzes course reviews using state-of-the-art NLP techniques to extract:
- ⭐ Overall sentiment (1–5 star rating)
- 😊 Emotion distribution
- ✅ What students liked
- ⚠️ Common complaints
- 📌 Aspect-based sentiment insights (e.g., content, instructor, pacing)
- 📈 Side-by-side comparison of two courses

Built with `Streamlit`, `transformers`, and custom NLP pipelines.

---

## 🚀 Features

- 💬 **Sentiment Analysis** – Predicts star ratings from text reviews
- 🔥 **Emotion Detection** – Maps reviews to fine-grained emotions (e.g., joy, anger)
- ✨ **Summarization** – Auto-generates likes and complaints from detected emotions
- 🧠 **Aspect-Based Sentiment Analysis (ABSA)** – Evaluates opinions on content, instructor, pacing, etc.
- 📊 **Course Comparison** – Compare two courses with side-by-side analysis
- 🎨 **Interactive UI** – Streamlit interface with visualizations and summaries

---

## 🛠️ Tech Stack

- `Python 3.10+`
- `Streamlit`
- `transformers` (BERT, GoEmotions model)
- `matplotlib`, `seaborn` (visualizations)
- `scikit-learn`, `pandas` (processing & metrics)

---

## ⚙️ Setup Instructions

```bash
# Clone the repo
git clone https://github.com/your-username/e-learning-review-analyzer.git
cd e-learning-review-analyzer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
------------------------
conda env create -f environment.yml  # Recreates exact environment
conda activate rev-env
streamlit run app.py

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py

🧪 Sample Input
Enter reviews like:
This course had great content and a very clear instructor. But the pace was a bit slow and I lost interest.
I loved the hands-on projects, they really helped me understand the concepts.
Too fast-paced for beginners, I felt lost after the first few lectures.
Excellent content, but I wish there were more real-world examples.
The quizzes were confusing and poorly worded.
Great introduction to the topic, and the visuals were very helpful.
I didn't like the instructor's teaching style, it was hard to stay focused.
The course was okay, but it needs more advanced topics.
Amazing course! I feel much more confident in this subject now.
Some sections were repetitive and could have been condensed.
This course was incredibly well-structured and the instructor explained everything clearly.


# If Module torch not found
pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 torchaudio==2.0.0+cu117 -f https://download.pytorch.org/whl/torch_stable.html
