<div align="center">

# 🐦 SocialSway

### Twitter Sentiment Analysis Using Machine Learning

**Turning tweets into insights with Natural Language Processing.**

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python\&logoColor=white)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--learn-orange)](https://scikit-learn.org/)
[![NLP](https://img.shields.io/badge/AI-Natural%20Language%20Processing-purple)](https://en.wikipedia.org/wiki/Natural_language_processing)
[![Status](https://img.shields.io/badge/Project-Completed-success)](#conclusion)

*Understand the emotions behind social media conversations.*

</div>

---

## 📌 Table of Contents

* [🌟 About the Project](#-about-the-project)
* [✨ Key Features](#-key-features)
* [🛠️ Technologies Used](#️-technologies-used)
* [⚙️ How It Works](#️-how-it-works)
* [📂 Dataset](#-dataset)
* [📊 Model Evaluation](#-model-evaluation)
* [📈 Visualizations](#-visualizations)
* [🚀 Getting Started](#-getting-started)
* [📁 Project Structure](#-project-structure)
* [👩‍💻 My Role](#-my-role)
* [🎯 Conclusion](#-conclusion)

---

## 🌟 About the Project

**SocialSway** is a machine learning project that analyzes tweets and identifies the sentiment expressed in social media text.

Using **Natural Language Processing (NLP)**, **TF-IDF vectorization**, and **Logistic Regression**, the project processes tweets and classifies their sentiment into positive, negative, or neutral categories, depending on the implemented label mapping.

The project demonstrates an end-to-end machine learning workflow, from text preprocessing and feature extraction to model training, evaluation, and visualization.

### 💡 Why SocialSway?

Social media contains millions of opinions, reactions, and conversations. Analyzing these messages can help reveal public sentiment and understand how people respond to different topics.

SocialSway explores how machine learning can turn unstructured text into meaningful sentiment insights.

---

## ✨ Key Features

| Feature                         | Description                                                                                      |
| ------------------------------- | ------------------------------------------------------------------------------------------------ |
| 🧹 **Text Preprocessing**       | Cleans tweets by removing URLs, mentions, punctuation, and other unwanted text.                  |
| 🔢 **TF-IDF Vectorization**     | Converts textual data into numerical features suitable for machine learning.                     |
| 🤖 **Sentiment Classification** | Uses Logistic Regression to predict sentiment from tweet text.                                   |
| 📊 **Model Evaluation**         | Evaluates predictions using accuracy, precision, recall, F1-score, and a confusion matrix.       |
| ☁️ **Word Clouds**              | Visualizes frequently occurring words in sentiment-related text.                                 |
| 📈 **Data Visualization**       | Generates sentiment distribution charts and confusion matrix heatmaps.                           |
| 💬 **Interactive Prediction**   | Allows users to enter text and explore its predicted sentiment, if supported by the application. |

---

## 🛠️ Technologies Used

| Technology                     | Purpose                                  |
| ------------------------------ | ---------------------------------------- |
| 🐍 Python                      | Core programming language                |
| 🧠 Natural Language Processing | Text processing and sentiment analysis   |
| 📚 Scikit-learn                | Machine learning and model evaluation    |
| 🔢 TF-IDF                      | Text feature extraction                  |
| 🤖 Logistic Regression         | Sentiment classification                 |
| 🧹 Text Preprocessing          | Cleaning and normalizing tweets          |
| 📊 Matplotlib / Seaborn        | Data visualization, where implemented    |
| ☁️ WordCloud                   | Word cloud generation, where implemented |
| 🗃️ Sentiment140               | Twitter sentiment dataset                |

---

## ⚙️ How It Works

SocialSway follows a structured machine learning pipeline:

```text
       ┌────────────────────────┐
       │   Sentiment140 Dataset │
       └────────────┬───────────┘
                    ▼
       ┌────────────────────────┐
       │   Text Preprocessing   │
       │ Cleaning and Filtering │
       └────────────┬───────────┘
                    ▼
       ┌────────────────────────┐
       │   TF-IDF Vectorization │
       │ Text → Numerical Data  │
       └────────────┬───────────┘
                    ▼
       ┌────────────────────────┐
       │   Logistic Regression  │
       │     Model Training     │
       └────────────┬───────────┘
                    ▼
       ┌────────────────────────┐
       │   Model Evaluation     │
       │ Metrics and Analysis   │
       └────────────┬───────────┘
                    ▼
       ┌────────────────────────┐
       │ Visualizations and     │
       │ Sentiment Prediction   │
       └────────────────────────┘
```

### 🔄 Workflow Steps

1. **Data Collection:** Load the tweet dataset and corresponding sentiment labels.
2. **Text Preprocessing:** Clean tweet text to reduce noise and standardize the input.
3. **Feature Extraction:** Apply TF-IDF to convert text into numerical feature vectors.
4. **Model Training:** Train a Logistic Regression classifier using the prepared data.
5. **Model Evaluation:** Assess model performance using classification metrics.
6. **Visualization:** Display sentiment distributions and other evaluation visualizations.
7. **Prediction:** Use the trained model to classify new text, if interactive prediction is implemented.

---

## 📂 Dataset

SocialSway uses the **Sentiment140 dataset**, a collection of tweets labeled according to sentiment.

The project uses the following dataset files locally:

* **Training dataset:** `training.1600000.processed.noemoticon.csv`
* **Testing dataset:** `testdata.manual.2009.06.14.csv`

The training dataset is used to develop the sentiment classification model, while the testing dataset can be used to evaluate predictions.

> **Note:** The datasets are not included in this GitHub repository because the training file exceeds GitHub's standard file-size limit. Place the dataset files in the appropriate local project directory before running the application.

---

## 📊 Model Evaluation

SocialSway uses standard classification metrics to evaluate the Logistic Regression model.

| Metric               | What It Measures                                                           |
| -------------------- | -------------------------------------------------------------------------- |
| **Accuracy**         | The proportion of predictions that are correct.                            |
| **Precision**        | How many predicted positive cases are actually positive for a given class. |
| **Recall**           | How many actual cases of a class the model correctly identifies.           |
| **F1-Score**         | The harmonic mean of precision and recall.                                 |
| **Confusion Matrix** | A breakdown of correct and incorrect predictions for each class.           |

These metrics help assess the model's performance and identify areas for improvement.

*Actual evaluation scores can be added after running the model.*

---

## 📈 Visualizations

The project supports visual analysis of sentiment and model predictions through visualizations such as:

* 📊 **Sentiment Distribution Charts:** Explore the distribution of sentiment labels.
* ☁️ **Word Clouds:** Highlight frequently occurring words in tweet text.
* 🔥 **Confusion Matrix Heatmaps:** Examine classification performance across classes.
* 💬 **Text-Based Sentiment Visualization:** Explore the sentiment predicted for user-entered text, where implemented.

*Screenshots of actual project outputs can be added here to showcase the results.*

---

## 🚀 Getting Started

Follow these steps to set up SocialSway on your local machine.

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/anikhaaiman01/SocialSway-Twitter-Sentiment-Analysis.git
cd SocialSway-Twitter-Sentiment-Analysis
```

### 2️⃣ Set Up a Virtual Environment (Recommended)

```bash
python -m venv venv
```

**Windows:**

```bash
venv\Scripts\activate
```

### 3️⃣ Install Dependencies

If your repository contains a `requirements.txt` file:

```bash
pip install -r requirements.txt
```

Otherwise, install the libraries required by your implementation. A typical setup may include:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn wordcloud nltk
```

You may omit libraries that your code does not use.

### 4️⃣ Add the Dataset

Download the Sentiment140 dataset from its original source and place the required CSV files in the locations expected by your code.

Ensure that the dataset filenames and paths match those configured in `main.py`.

### 5️⃣ Run the Application

```bash
python main.py
```

Follow any prompts displayed by the program.

> **Troubleshooting:** If the program reports a missing module or file, check the installed dependencies and dataset paths.

---

## 📁 Project Structure

The repository currently contains the following main files and directories:

```text
SocialSway-Twitter-Sentiment-Analysis/
│
├── .vscode/          # VS Code settings
├── .gitignore        # Files excluded from Git
├── README.md         # Project documentation
└── main.py           # Main Python application
```

Your project may contain additional files locally. Update this structure if you add files such as `requirements.txt`, screenshots, or separate source modules.

---

## 👩‍💻 My Role

I developed **SocialSway** as part of the *Machine Learning Techniques* course, focusing on implementing a machine learning-based sentiment analysis pipeline.

My work involved exploring text preprocessing, TF-IDF feature extraction, Logistic Regression classification, model evaluation, and visualization techniques for analyzing sentiment in social media text.

---

## 🎯 Conclusion

SocialSway demonstrates the practical application of **Machine Learning and Natural Language Processing** to social media sentiment analysis.

By combining text preprocessing, TF-IDF feature extraction, Logistic Regression, evaluation metrics, and data visualization, the project provides a foundation for exploring how machine learning can extract sentiment-related insights from unstructured text.

The project also provides opportunities for further development, including model optimization, improved text preprocessing, expanded datasets, and more advanced sentiment classification methods.

---

<div align="center">

### 💙 Thank you for visiting SocialSway!

**Explore the code. Understand the sentiment. Discover the insights.**

Made with Python and Machine Learning.

</div>
