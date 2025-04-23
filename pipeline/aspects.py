import re
from pipeline.sentiment_analysis import predict_sentiment_for_sentences

# Aspect keywords

# aspects_keywords= {
#     "content_quality": [
#         "content", "material", "curriculum", "topics", "depth", "up-to-date", 
#         "outdated", "comprehensive", "shallow", "missing", "coverage", "syllabus"
#     ],
#     "instructor": [
#         "instructor", "teacher", "lecturer", "explanation", "teach", "presentation",
#         "communication", "engaging", "monotone", "knowledgeable", "approachable", "feedback"
#     ],
#     "pacing": [
#         "pace", "pacing", "speed", "fast", "slow", "rushed", "overwhelming",
#         "lag", "drag", "boring", "repetitive", "too much", "too little"
#     ],
#     "assignments": [
#         "assignment", "exercise", "project", "quiz", "exam", "test", "homework",
#         "challenge", "difficulty", "grading", "rubric", "feedback", "solution"
#     ],
#     "platform_experience": [
#         "platform", "website", "interface", "UI", "UX", "bug", "glitch", "laggy",
#         "load", "crash", "mobile", "responsive", "video", "audio", "subtitle"
#     ]
# }

aspects_keywords= {
    "content_quality": ["content", "material", "topics", "lecture", "information"],
    "instructor": ["teacher", "instructor", "professor", "explained", "taught"],
    "pacing": ["pace", "speed", "fast", "slow", "rushed", "timing"],
    "assignments": ["assignment", "quiz", "homework", "exercise", "task"],
    "difficulty": ["easy", "difficult", "challenging", "hard", "simple"],
    "platform_experience": ["website", "platform", "interface", "navigation", "bugs"],
}
# ASPECT_KEYWORDS = {
#     "content_quality": [
#         "content", "material", "curriculum", "topics", "depth", "up-to-date", 
#         "outdated", "comprehensive", "shallow", "missing", "coverage", "syllabus"
#     ],
#     "instructor": [
#         "instructor", "teacher", "lecturer", "explanation", "teach", "presentation",
#         "communication", "engaging", "monotone", "knowledgeable", "approachable", "feedback"
#     ],
#     "pacing": [
#         "pace", "pacing", "speed", "fast", "slow", "rushed", "overwhelming",
#         "lag", "drag", "boring", "repetitive", "too much", "too little"
#     ],
#     "assignments": [
#         "assignment", "exercise", "project", "quiz", "exam", "test", "homework",
#         "challenge", "difficulty", "grading", "rubric", "feedback", "solution"
#     ],
#     "platform_experience": [
#         "platform", "website", "interface", "UI", "UX", "bug", "glitch", "laggy",
#         "load", "crash", "mobile", "responsive", "video", "audio", "subtitle"
#     ],
#     "practical_skills": [
#         "hands-on", "practical", "real-world", "portfolio", "job", "career",
#         "application", "example", "case study", "useful", "relevant", "hireable"
#     ],
#     "support_resources": [
#         "support", "help", "Q&A", "forum", "community", "slack", "discord",
#         "response", "mentor", "TA", "office hours", "documentation", "resource"
#     ]
    # "web_tech_stack": [
    #     "HTML", "CSS", "JavaScript", "React", "Node", "API", "DOM", "Redux", 
    #     "Express", "MongoDB", "SQL", "Firebase", "Bootstrap"
    # ],
    # "python_tech_stack": [
    #     "Python", "Django", "Flask", "Pandas", "NumPy", "OOP", "lambda", 
    #     "decorators", "generators", "async", "PyTorch", "matplotlib", "Selenium"
    # ],
    #     "general_tech": [
    #     "Java", "C++", "SQL", "NoSQL", "AWS", "Docker", "Kubernetes", "cloud", 
    #     "Azure", "CI/CD", "Linux", "terminal", "command line", "Agile", "DevOps",
    #     "debugging", "algorithm", "data structure", "Git", "GitHub", "VS Code", "IDE"
    # ]
# }

def split_into_sentences(text):
    return re.split(r'(?<=[.!?])\s+', text.strip())

def run_aspect_sentiment_analysis(reviews, model_path="Jarvis8191/sentiment-model"):
    aspect_results = {aspect: [] for aspect in aspects_keywords}
    all_sentences = []
    sentence_map = []

    for review_idx, review in enumerate(reviews):
        sentences = split_into_sentences(review)
        for sentence in sentences:
            all_sentences.append(sentence)
            sentence_map.append((review_idx, sentence))

    sentiments = predict_sentiment_for_sentences(all_sentences, model_path)

    for i, (review_idx, sentence) in enumerate(sentence_map):
        label = sentiments[i]
        for aspect, keywords in aspects_keywords.items():
            if any(kw in sentence.lower() for kw in keywords):
                aspect_results[aspect].append((sentence, label))
                break

    # ✅ Remove near-duplicates by normalizing casing and punctuation
    for aspect in aspect_results:
        seen = set()
        unique_entries = []
        for sentence, label in aspect_results[aspect]:
            normalized = re.sub(r'[^\w\s]', '', sentence.lower())  # remove punctuation, lowercase
            key = (normalized.strip(), label)
            if key not in seen:
                seen.add(key)
                unique_entries.append((sentence, label))
        aspect_results[aspect] = unique_entries

    return aspect_results

