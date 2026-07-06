# 📝 AI-Powered Plagiarism Detection System

An advanced plagiarism detection system built using **Python** and **Natural Language Processing (NLP)**. The project detects textual similarity between documents using a combination of **TF-IDF Vectorization**, **Cosine Similarity**, **Fuzzy String Matching**, and **text preprocessing techniques** to provide more accurate plagiarism detection.

---

## 📌 Features

- Detects plagiarism between multiple text documents
- NLP-based text preprocessing
- Removes punctuation, numbers, URLs, and stopwords
- Performs lemmatization using NLTK
- Uses TF-IDF Vectorization with Uni-grams and Bi-grams
- Calculates Cosine Similarity
- Calculates Fuzzy Similarity using RapidFuzz
- Combines multiple similarity metrics into a final plagiarism score
- Classifies plagiarism into different risk levels
- Generates plagiarism reports in CSV format
- Supports both small text files and large article datasets

---

## 🛠 Technologies Used

- Python
- Scikit-learn
- NLTK
- RapidFuzz
- Pandas

---

## 📂 Project Structure

```
Plagiarism-checker-Python/
│
├── app.py
├── Articles.csv
├── fatma.txt
├── john.txt
├── juma.txt
├── plagiarism_report.csv
├── news_plagiarism_report.csv
├── requirements.txt
├── README.md
└── pictures/
```

---

## ⚙️ How It Works

### Step 1 – Text Preprocessing

Each document is cleaned using NLP techniques:

- Convert text to lowercase
- Remove URLs
- Remove numbers
- Remove punctuation
- Tokenize text
- Remove stopwords
- Lemmatize words

Example:

Original:

```
Python is AMAZING!! I love learning Python because it is powerful.
```

Processed:

```
python amaze love learn python powerful
```

---

### Step 2 – TF-IDF Vectorization

The cleaned documents are converted into numerical vectors using **TF-IDF Vectorizer**.

Configuration:

- Lowercase conversion
- Uni-grams and Bi-grams
- Sublinear TF scaling
- Document frequency filtering

---

### Step 3 – Cosine Similarity

Cosine Similarity measures how similar two document vectors are.

Range:

- 0 → Completely different
- 1 → Identical

---

### Step 4 – Fuzzy String Matching

RapidFuzz compares the textual similarity between processed documents and produces another similarity score.

---

### Step 5 – Final Similarity Score

Both similarity metrics are combined into a weighted score.

```
Final Score = (0.7 × Cosine Similarity) + (0.3 × Fuzzy Similarity)
```

Cosine Similarity receives a higher weight because it captures semantic similarity more effectively than simple string matching.

---

### Step 6 – Plagiarism Classification

| Final Score | Status |
|-------------|--------|
| 0.80 – 1.00 | 🔴 High Plagiarism |
| 0.60 – 0.79 | 🟠 Possible Plagiarism |
| 0.40 – 0.59 | 🟡 Needs Review |
| 0.00 – 0.39 | 🟢 Safe |

---

## 📊 Sample Output

```
======================================================================
Document 1        : john.txt
Document 2        : juma.txt

Cosine Similarity : 0.44
Fuzzy Similarity  : 0.71

Final Score       : 0.52

Status            : 🟡 Needs Review
```

---

## 📈 Output Reports

The project automatically generates:

### Student Document Report

```
plagiarism_report.csv
```

Contains:

- Document Name
- Cosine Similarity
- Fuzzy Similarity
- Final Score
- Plagiarism Status

---

### News Dataset Report

```
news_plagiarism_report.csv
```

Contains:

- Article Number
- Article Headings
- Cosine Similarity
- Fuzzy Similarity
- Final Score
- Classification

---

## 📂 Dataset

The project has been tested on:

### Student Documents

```
fatma.txt
john.txt
juma.txt
```

### News Articles Dataset

```
Articles.csv
```

A random sample of 50 news articles is selected to demonstrate plagiarism detection on larger datasets.

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Plagiarism-checker-Python.git

cd Plagiarism-checker-Python
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project

Execute:

```bash
python app.py
```

Or run the Jupyter Notebook / Google Colab notebook for the enhanced NLP pipeline.

---

## 📚 Future Improvements

- Semantic similarity using Sentence Transformers (BERT)
- Named Entity Recognition (NER)
- PDF and DOCX support
- Web interface using Flask or Streamlit
- Deep Learning-based plagiarism detection
- Highlight copied sentences
- Upload multiple files through GUI
- Database integration for document storage

---

## 👨‍💻 Author 

Developed as an NLP and Machine Learning project using Python to demonstrate practical plagiarism detection through text preprocessing, feature extraction, similarity measurement, and automated report generation.