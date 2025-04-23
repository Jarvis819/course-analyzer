import ollama

def generate_summary(reviews, model='phi3'):
    content = "Here are some course reviews:\n\n" + "\n".join(reviews[:20])
    content += "\n\nPlease give a short paragraph summary of this course."

    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": content}]
    )

    return response['message']['content']



def answer_user_question(reviews, user_question, model='phi3'):
    context = "\n".join(reviews[:20])
    prompt = f"""
You are an AI assistant that helps answer questions about course reviews.

Here are the reviews:
{context}

User question:
{user_question}
""".strip()

    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )

    return response['message']['content']
