# Install required libraries (run this in a notebook or terminal, not in a .py file)
# !pip install nltk scikit-learn matplotlib seaborn wordcloud

# Import libraries
import importlib
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Resolve optional dependencies safely so the project does not fail when packages are missing.
try:
    sns = importlib.import_module('seaborn')
except ModuleNotFoundError:
    sns = None

try:
    from wordcloud import WordCloud  # type: ignore[reportMissingImports]
except ModuleNotFoundError:
    WordCloud = None

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Resolve NLTK safely so the project does not fail when the package is missing.
try:
    nltk = importlib.import_module('nltk')
    stopwords_module = importlib.import_module('nltk.corpus')
    stopwords = stopwords_module.stopwords
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))
except ModuleNotFoundError:
    nltk = None
    stop_words = {
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
        'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself',
        'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
        'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be',
        'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
        'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for',
        'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above',
        'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',
        'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each',
        'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
        'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'
    }

# -------------------------------
# Stage 1: Data Collection
# -------------------------------
df = pd.read_csv("training.1600000.processed.noemoticon.csv",
                 encoding='latin-1',
                 names=['target','id','date','flag','user','text'])

df = df[['target','text']]  # keep only sentiment + text

# Map labels: 0 = Negative, 2 = Neutral, 4 = Positive → 1
df['target'] = df['target'].replace({0:0, 2:2, 4:1})

print("Label distribution:\n", df['target'].value_counts())

# -------------------------------
# Stage 2: Preprocessing
# -------------------------------
def preprocess_tweet(tweet):
    tweet = tweet.lower()  # lowercase
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet)  # remove URLs
    tweet = re.sub(r'@\w+|#','', tweet)  # remove mentions and hashtags
    tweet = re.sub(r'[^A-Za-z\s]', '', tweet)  # remove punctuation/numbers
    tokens = tweet.split()
    tokens = [word for word in tokens if word not in stop_words]  # remove stopwords
    return " ".join(tokens)

df['clean_text'] = df['text'].apply(preprocess_tweet)

# Drop empty tweets after cleaning
df = df[df['clean_text'].str.strip() != ""]

print("\nPreprocessed sample:\n", df['clean_text'].head())

# -------------------------------
# Stage 3: Feature Extraction
# -------------------------------
# Stratified split ensures all classes appear in train/test
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['target'],
    test_size=0.2,
    random_state=42,
    stratify=df['target']
)

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -------------------------------
# Stage 4: Model Training
# -------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# -------------------------------
# Stage 5: Evaluation
# -------------------------------
y_pred = model.predict(X_test_vec)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# -------------------------------
# Stage 6: Visualization
# -------------------------------
# Sentiment Distribution Chart
sns.countplot(x=df['target'])
plt.title("Sentiment Distribution in Dataset")
plt.xlabel("Sentiment (0=Negative, 1=Positive, 2=Neutral)")
plt.ylabel("Count")
plt.show()

# Safe Word Cloud Function
def safe_wordcloud(texts, title, bg_color="white"):
    if len(texts.strip()) == 0:
        print(f"Skipping {title} word cloud (no words).")
        return
    wc = WordCloud(width=800, height=400, background_color=bg_color).generate(texts)
    plt.figure(figsize=(10,5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.show()

# Generate safely
safe_wordcloud(" ".join(df[df['target']==1]['clean_text']), "Positive Tweets Word Cloud", "white")
safe_wordcloud(" ".join(df[df['target']==0]['clean_text']), "Negative Tweets Word Cloud", "black")
safe_wordcloud(" ".join(df[df['target']==2]['clean_text']), "Neutral Tweets Word Cloud", "gray")

# -------------------------------
# Stage 7: Prediction
# -------------------------------
def predict_sentiment(new_tweet):
    clean = preprocess_tweet(new_tweet)
    vec = vectorizer.transform([clean])
    pred = model.predict(vec)[0]
    if pred == 1:
        return "Positive"
    elif pred == 0:
        return "Negative"
    else:
        return "Neutral"
# -------------------------------
# Stage 8: Interactive Input + Word Cloud
# -------------------------------
def predict_and_wordcloud(new_tweet):
    clean = preprocess_tweet(new_tweet)
    vec = vectorizer.transform([clean])
    pred = model.predict(vec)[0]

    if pred == 1:
        sentiment = "Positive"
        bg_color = "white"
    elif pred == 0:
        sentiment = "Negative"
        bg_color = "black"
    else:
        sentiment = "Neutral"
        bg_color = "gray"

    print(f"\nInput: {new_tweet}")
    print(f"Preprocessed: {clean}")
    print(f"Predicted Sentiment: {sentiment}")

    # Word cloud for the input statement
    if len(clean.strip()) > 0:
        wc = WordCloud(width=800, height=400, background_color=bg_color).generate(clean)
        plt.figure(figsize=(8,4))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title(f"{sentiment} Word Cloud")
        plt.show()
    else:
        print("No words available for word cloud.")

# Ask user for input
user_input = input("Enter a sentence to analyze: ")
predict_and_wordcloud(user_input)
