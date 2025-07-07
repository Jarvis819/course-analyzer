import streamlit as st
import pandas as pd 
import matplotlib.pyplot as plt
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch

from pipeline.sentiment_analysis import analyze_sentiment
from pipeline.summarization import generate_summary
from pipeline.emotion import detect_emotions_from_list
from pipeline.aspects import run_aspect_sentiment_analysis
from utils.utils import (
    categorize_emotion_batches,
    get_emotion_distribution_plot,
    get_sentiment_distribution_plot,
    clean_and_capitalize_sentences,
)

# Load the Hugging Face model (only once)
# @st.cache_resource
# def load_hf_model():
#     return pipeline("text-generation", model="microsoft/DialoGPT-medium",  device=0 if torch.cuda.is_available() else -1)
# @st.cache_resource
# def load_hf_model():
#     model_id = "microsoft/DialoGPT-small"
#     tokenizer = AutoTokenizer.from_pretrained(model_id)
#     model = AutoModelForCausalLM.from_pretrained(model_id).to("cpu")  
#     return tokenizer, model
def load_hf_model():
    model_id = "microsoft/DialoGPT-small"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).to("cpu")
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)
    return pipe
    
hf_pipeline = load_hf_model()


# Page Setup
st.set_page_config(page_title="E-learning Review Analyzer", layout="centered")
st.sidebar.title("📚 Navigation")
page = st.sidebar.radio("Go to:", ["Home", "Comparison Results", "AI-Powered Q&A"])


# Helper function to render results
def render_course_results(title, data):
    st.subheader(title)

    # Overall Sentiment
    st.markdown("### 📈 Overall Sentiment")
    score = data['sentiments']
    progress_html = f"""
    <div style="background-color: #ddd; border-radius: 10px; height: 30px; width: 100%;">
    <div style="background-color: #0A76F6; width: {score}%; height: 100%; border-radius: 10px;"></div>
    </div>
    <div style="text-align: center; font-size: 18px; margin-top: 8px;">
    <strong>{score:.2f}% Positive</strong>
    </div>
    """
    st.markdown(progress_html, unsafe_allow_html=True)
    st.markdown("### 🧾 Sentiment Breakdown")
    st.pyplot(data["sentiment_chart"])

    # Emotion Chart
    st.markdown("### 🎭 Emotion Breakdown")
    emotion_chart = get_emotion_distribution_plot(data['emotions'])
    st.pyplot(emotion_chart)

    # Summary
    st.markdown("### 🧠 AI-generated Summary")
    st.markdown("**✅ What Students Liked**")
    st.markdown(f"> {data['liked_summary']}")
    st.markdown("**⚠️ Common Complaints**")
    st.markdown(f"> {data['complaint_summary']}")

    # Aspects
    st.markdown("### 🧩 Aspect-Based Sentiment Analysis")
    for aspect, entries in data["aspect_sentiments"].items():
        if entries:
            st.markdown(f"**🔹 {aspect.replace('_', ' ').title()}**")
            for sent, label in entries[:5]:
                st.markdown(f"> {'⭐️' * (label + 1)} {sent}")
    st.markdown("---")


# --------------------- HOME PAGE ---------------------
if page == "Home":
    st.title("📊 E-learning Review Analyzer")
    if "course1_results" not in st.session_state: 
        st.markdown("Choose how you want to provide course reviews:")

        input_mode = st.radio("Select input method:", ["📝 Paste Text", "📁 Upload CSV"])
        reviews = []
        
        if input_mode == "📝 Paste Text":
            review_input = st.text_area("Paste course reviews below (separate each by a new line):", height=250)
            if review_input.strip():
                raw_reviews = [r.strip() for r in review_input.strip().split('\n') if r.strip()]
                unique_reviews = list(dict.fromkeys(raw_reviews))  # removes duplicates while preserving order
                if len(unique_reviews) > 500:
                    st.warning("⚠️ Only the first 500 unique reviews will be analyzed.")
                reviews = unique_reviews[:500]



        elif input_mode == "📁 Upload CSV":
            uploaded_file = st.file_uploader("Upload a CSV file with a 'review' column", type=["csv"], key="file1")
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                df = df.drop_duplicates(subset=["review"])
                if 'review' not in df.columns:
                    st.error("❌ The uploaded CSV must contain a column named 'review'.")
                else:
                    st.success(f"✅ Uploaded {len(df)} rows.")
                    # Limit to first N rows if too large
                    max_reviews = 500  # Change this if desired
                    if len(df) > max_reviews:
                        st.warning(f"File has more than {max_reviews} reviews. Only analyzing the first {max_reviews}.")
                        df = df.head(max_reviews)
                    reviews = df['review'].fillna("").tolist()

        if st.button("Analyze", key="analyze_course1"):
            if not reviews:
                st.warning("⚠️ Please provide at least one valid review.")
            else:
                with st.spinner("Analyzing..."):
                    sentiments, score = analyze_sentiment(reviews)
                    emotions = detect_emotions_from_list(reviews)
                    liked, complaints = categorize_emotion_batches(reviews, emotions)
                    liked_summary, complaint_summary = generate_summary(liked, complaints)
                    aspect_sentiments = run_aspect_sentiment_analysis(reviews)
                    sentiment_chart = get_sentiment_distribution_plot(sentiments)
                    st.session_state["course1_results"] = {
                        "sentiments": score,
                        "emotions": emotions,
                        "aspect_sentiments": aspect_sentiments,
                        "liked_summary": clean_and_capitalize_sentences(liked_summary),
                        "complaint_summary": clean_and_capitalize_sentences(complaint_summary),
                        "liked":liked,
                        "sentiment_chart": sentiment_chart,
                        "complaints": complaints
                        
                    }
                    st.session_state.compare_mode = False  # Reset

    # Show results if available
    if "course1_results" in st.session_state:
        render_course_results("📘 Course 1 Analysis", st.session_state["course1_results"])

        if not st.session_state.get("compare_mode"):
            if st.button("🔁 Compare with Another Course"):
                st.session_state.compare_mode = True

    # Second Course Entry
    if "course2_results" not in st.session_state:
        if st.session_state.get("compare_mode", False) or "course2_results" in st.session_state:
            st.markdown("---")
            st.subheader("🆚 Course 2 Input")

            # Always show input interface in compare mode
            second_input_mode = st.radio("Select input method for Course 2:", ["📝 Paste Text", "📁 Upload CSV"], key="mode2")
            reviews2 = []
            
            if second_input_mode == "📝 Paste Text":
                review_input2 = st.text_area("Paste course reviews below (separate each by a new line):", height=250)
                if review_input2.strip():
                    raw_reviews2 = [r.strip() for r in review_input2.strip().split('\n') if r.strip()]
                    unique_reviews2 = list(dict.fromkeys(raw_reviews2))  # removes duplicates while preserving order
                    if len(unique_reviews2) > 500:
                        st.warning("⚠️ Only the first 500 unique reviews will be analyzed.")
                    reviews2 = unique_reviews2[:500]

            elif second_input_mode == "📁 Upload CSV":
                uploaded_file2 = st.file_uploader("Upload second course CSV (with 'review' column)", type=["csv"], key="file2")
                if uploaded_file2 is not None:
                    df2 = pd.read_csv(uploaded_file2)
                    df2 = df2.drop_duplicates(subset=["review"])
                    if 'review' not in df2.columns:
                        st.error("❌ The uploaded CSV must contain a column named 'review'.")
                    else:
                        st.success(f"✅ Uploaded {len(df2)} rows.")    
                        # Limit to first N rows if too large
                        max_reviews = 500  # Change this if desired
                        if len(df2) > max_reviews:
                            st.warning(f"File has more than {max_reviews} reviews. Only analyzing the first {max_reviews}.")
                            df2 = df2.head(max_reviews)
                        reviews2 = df2['review'].fillna("").tolist()

            if st.button("Analyze Second Course"):
                if not reviews2:
                    st.warning("⚠️ Please provide at least one valid review.")
                else:
                    with st.spinner("Analyzing second course..."):
                        sentiments2, sentiment_score2 = analyze_sentiment(reviews2)
                        detected_emotions2 = detect_emotions_from_list(reviews2)
                        liked2, complaints2 = categorize_emotion_batches(reviews2, detected_emotions2)
                        liked_summary2, complaint_summary2 = generate_summary(liked2, complaints2)
                        liked_summary2 = clean_and_capitalize_sentences(liked_summary2)
                        complaint_summary2 = clean_and_capitalize_sentences(complaint_summary2)
                        aspect_sentiments2 = run_aspect_sentiment_analysis(reviews2)
                        emotion_chart2 = get_emotion_distribution_plot(detected_emotions2)
                        sentiment_chart2 = get_sentiment_distribution_plot(sentiments2, bar_color="#00B894")

                        st.session_state["course2_results"] = {
                            "sentiments": sentiment_score2,
                            "emotions": detected_emotions2,
                            "emotion_chart": emotion_chart2,
                            "liked_summary": liked_summary2,
                            "complaint_summary": complaint_summary2,
                            "aspect_sentiments": aspect_sentiments2,
                            "sentiment_chart": sentiment_chart2,
                            "liked2": liked2,
                            "complaints2": complaints2
                        }
                        st.session_state["compare_mode"] = True
                        st.success("Course 2 analyzed! You can now view both on the Comparison Results page.")

    # 🔁 Always show Course 2 results if available
    if "course2_results" in st.session_state:
        results2 = st.session_state["course2_results"]
        st.markdown("---")
        st.subheader("📈 Course 2: Overall Sentiment")
        progress_html2 = f"""
        <div style="background-color: #ddd; border-radius: 10px; height: 30px; width: 100%;">
        <div style="background-color: #00B894; width: {results2['sentiments']}%; height: 100%; border-radius: 10px;"></div>
        </div>
        <div style="text-align: center; font-size: 18px; margin-top: 8px;">
        <strong>{results2['sentiments']:.2f}% Positive</strong>
        </div>
        """
        st.markdown(progress_html2, unsafe_allow_html=True)
        st.pyplot(results2['sentiment_chart'])

        st.subheader("🎭 Emotion Breakdown (Course 2)")
        st.pyplot(results2['emotion_chart'])

        st.subheader("🧠 AI-generated Summary (Course 2)")
        st.markdown("**✅ What Students Liked**")
        st.markdown(f"> {results2['liked_summary']}")
        st.markdown("**⚠️ Common Complaints**")
        st.markdown(f"> {results2['complaint_summary']}")

        st.subheader("🧩 Aspect-Based Sentiment Analysis (Course 2)")
        for aspect, entries in results2["aspect_sentiments"].items():
            if entries:
                st.markdown(f"**🔹 {aspect.replace('_', ' ').title()}**")
                for sent, label in entries[:5]:
                    st.markdown(f"> {'⭐️' * (label + 1)} {sent}")


# --------------------- COMPARISON PAGE ---------------------
elif page == "Comparison Results":
    st.title("📊 Course Comparison Results")
    if "course1_results" not in st.session_state or "course2_results" not in st.session_state:
        st.warning("⚠️ Please analyze both courses from the Home tab first.")
    else:
        from utils.comparisons import show_comparison
        show_comparison(st.session_state["course1_results"], st.session_state["course2_results"])


# --------------------- AI Q&A PAGE ---------------------
elif page == "AI-Powered Q&A":
    st.title("🧠 AI-Powered Q&A on Course Reviews")

    if "course1_results" not in st.session_state:
        st.warning("⚠️ Please analyze at least one course on the Home page first.")
    else:
        reviews = st.session_state["course1_results"]["liked"] + st.session_state["course1_results"]["complaints"]


        def ask_hf(question, reviews):
            prompt = f"""
        You're an assistant trained to answer questions about course reviews.
        Please answer the question concisely, using no more than 3 to 5 sentences.
        
Reviews:
{chr(10).join(reviews[:20])}

Question:
{question}
"""
            output = hf_pipeline(prompt, max_new_tokens=200, do_sample=True)[0]['generated_text']
            return output.replace(prompt.strip(), "").strip()



        with st.expander("📌 AI Summary Based on Reviews from Course 1"):
            if "ollama_summary" not in st.session_state:
                with st.spinner("Generating summary..."):
                    summary_prompt = "Summarize the sentiment and common opinions from these course reviews in 1 paragraph. Be concise."
                    st.session_state["ollama_summary"] = ask_hf(summary_prompt, reviews)
            
            st.markdown(f"**Summary:**\n\n{st.session_state['ollama_summary']}")

        if "course2_results" not in st.session_state:
            st.markdown("### 💬 Ask a Question")
            user_question = st.text_input("What do you want to know about this course?")
            if st.button("Ask AI"):
                if user_question.strip():
                    with st.spinner("Thinking..."):
                        ai_response = ask_hf(user_question, reviews)
                        st.markdown(f"**Answer:**\n\n{ai_response}")
                else:
                    st.warning("Please enter a question.")

# For course 2 ---------------------------------------------------------

# Ensure Course 1 and Course 2 reviews are correctly combined.
if "course1_results" in st.session_state:
    course1_reviews = st.session_state["course1_results"]["liked"] + st.session_state["course1_results"]["complaints"]
else:
    course1_reviews = []

if "course2_results" in st.session_state:
    course2_reviews = st.session_state["course2_results"]["liked2"] + st.session_state["course2_results"]["complaints2"]
else:
    course2_reviews = []

# Function to ask Ollama for comparison answers.
def ask_hf_comparison(question, course1_reviews, course2_reviews):
    prompt = f"""
You're an AI assistant trained to compare and answer student queries about course reviews.

📘 Course 1 Reviews:
{chr(10).join(course1_reviews[:10])}

📗 Course 2 Reviews:
{chr(10).join(course2_reviews[:10])}

💬 User Question:
{question}

Please provide a concise and helpful answer in 3 to 5 sentences.
"""
    output = hf_pipeline(prompt, max_new_tokens=250, do_sample=True)[0]['generated_text']
    return output.replace(prompt.strip(), "").strip()



# **AI-Powered Q&A Section**
if page == "AI-Powered Q&A":
    
    if "course1_results" not in st.session_state:
        st.warning("⚠️ Please analyze at least one course on the Home page first.")
    else:
        reviews = st.session_state["course1_results"]["liked_summary"] + st.session_state["course1_results"]["complaint_summary"]

        # Only generate Course 2 summary in the AI-Powered Q&A page
        if "course2_results" in st.session_state:
            with st.expander("📌 AI Summary Based on Reviews from both the Courses"):
                if "ollama_summary2" not in st.session_state:
                    with st.spinner("Generating summary..."):
                        summary_prompt = "Summarize the sentiment and common opinions from these course reviews in 1 paragraph. Be concise."
                        st.session_state["ollama_summary2"] = ask_hf_comparison(summary_prompt, course1_reviews, course2_reviews)
                
                st.markdown(f"**Summary:**\n\n{st.session_state['ollama_summary2']}")

            user_question = st.text_input("Ask about or compare the two courses:")
            if st.button("Ask Comparison AI"):
                if user_question.strip():
                    with st.spinner("Thinking..."):
                        ai_response = ask_hf_comparison(user_question, course1_reviews, course2_reviews)
                        st.markdown(f"**Answer:**\n\n{ai_response}")
