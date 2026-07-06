import os
import re
import string
import pandas as pd

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from rapidfuzz import fuzz


# -----------------------------
# NLP Setup
# -----------------------------
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


# -----------------------------
# Text Preprocessing
# -----------------------------
def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))

    words = word_tokenize(text)

    cleaned = []

    for word in words:
        if word not in stop_words and len(word) > 2:
            cleaned.append(
                lemmatizer.lemmatize(word, pos="v")
            )

    return " ".join(cleaned)


# -----------------------------
# Similarity Functions
# -----------------------------
def fuzzy_similarity(text1, text2):
    return fuzz.ratio(text1, text2) / 100


def plagiarism_status(score):

    if score >= 0.80:
        return "High Plagiarism"

    elif score >= 0.60:
        return "Possible Plagiarism"

    elif score >= 0.40:
        return "Needs Review"

    else:
        return "Safe"


def final_similarity(cosine_score, fuzzy_score):
    return round((0.7 * cosine_score) + (0.3 * fuzzy_score), 2)


# -----------------------------
# Student TXT Analysis
# -----------------------------
def analyze_student_documents():

    print("\n==============================")
    print("Student Document Analysis")
    print("==============================")

    student_files = [
        file for file in os.listdir()
        if file.endswith(".txt")
    ]

    documents = []

    for file in student_files:
        with open(file, encoding="utf-8") as f:
            documents.append(preprocess(f.read()))

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2)
    )

    matrix = vectorizer.fit_transform(documents)

    similarity = cosine_similarity(matrix)

    results = []

    for i in range(len(student_files)):
        for j in range(i + 1, len(student_files)):

            cosine_score = similarity[i][j]
            fuzzy_score = fuzzy_similarity(
                documents[i],
                documents[j]
            )

            score = final_similarity(
                cosine_score,
                fuzzy_score
            )

            results.append({
                "Document 1": student_files[i],
                "Document 2": student_files[j],
                "Cosine Similarity": round(cosine_score, 2),
                "Fuzzy Similarity": round(fuzzy_score, 2),
                "Final Score": score,
                "Status": plagiarism_status(score)
            })

    report = pd.DataFrame(results)

    report.to_csv(
        "plagiarism_report.csv",
        index=False
    )

    print(report)

    print("\nStudent report saved as plagiarism_report.csv")


# -----------------------------
# News Dataset Analysis
# -----------------------------
def analyze_news_articles():

    if not os.path.exists("Articles.csv"):
        print("\nArticles.csv not found. Skipping news analysis.")
        return

    print("\n==============================")
    print("News Article Analysis")
    print("==============================")

    df = pd.read_csv(
        "Articles.csv",
        encoding="latin1"
    )

    news_df = df.sample(
        n=50,
        random_state=42
    ).reset_index(drop=True)

    articles = [
        preprocess(article)
        for article in news_df["Article"]
    ]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_df=0.90,
        min_df=2,
        sublinear_tf=True
    )

    matrix = vectorizer.fit_transform(articles)

    similarity = cosine_similarity(matrix)

    results = []

    for i in range(len(articles)):
        for j in range(i + 1, len(articles)):

            cosine_score = similarity[i][j]

            fuzzy_score = fuzzy_similarity(
                articles[i],
                articles[j]
            )

            score = final_similarity(
                cosine_score,
                fuzzy_score
            )

            results.append({

                "Article 1": i,
                "Article 2": j,

                "Heading 1":
                news_df.iloc[i]["Heading"],

                "Heading 2":
                news_df.iloc[j]["Heading"],

                "Cosine Similarity":
                round(cosine_score, 2),

                "Fuzzy Similarity":
                round(fuzzy_score, 2),

                "Final Score":
                score,

                "Status":
                plagiarism_status(score)

            })

    report = pd.DataFrame(results)

    report = report.sort_values(
        by="Final Score",
        ascending=False
    )

    report.to_csv(
        "news_plagiarism_report.csv",
        index=False
    )

    print(report.head(10))

    print("\nNews report saved as news_plagiarism_report.csv")


# -----------------------------
# Main Function
# -----------------------------
def main():

    print("=" * 60)
    print("AI Plagiarism Detection System")
    print("=" * 60)

    analyze_student_documents()

    analyze_news_articles()

    print("\nAnalysis Completed Successfully.")


if __name__ == "__main__":
    main()