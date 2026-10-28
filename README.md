SocialSway — Twitter Sentiment Analysis

SocialSway is a machine learning project that analyzes tweets and classifies their sentiment into positive, negative, or neutral categories using Natural Language Processing (NLP), TF-IDF feature extraction, and Logistic Regression.

The project demonstrates an end-to-end machine learning workflow, including text preprocessing, model training, evaluation, and visualization.

Features

- Text Preprocessing: Cleans tweets by removing URLs, mentions, hashtags, punctuation, and stopwords.
- TF-IDF Vectorization: Converts text into numerical features for machine learning.
- Sentiment Classification: Uses Logistic Regression to predict sentiment.
- Model Evaluation: Evaluates performance using accuracy, precision, recall, F1-score, and a confusion matrix.
- Data Visualization: Generates sentiment distribution charts, word clouds, and confusion matrix heatmaps.
- Interactive Prediction: Accepts user-entered text and predicts its sentiment with a word cloud visualization.

Technologies Used

- Python
- Natural Language Processing (NLP)
- Scikit-learn
- TF-IDF Vectorization
- Logistic Regression
- Data Visualization
- Sentiment140 Dataset

Project Workflow

1. Data Collection: Load the Sentiment140 dataset and sentiment labels.
2. Text Preprocessing: Clean and normalize tweet text.
3. Feature Extraction: Transform text into TF-IDF vectors.
4. Model Training: Train a Logistic Regression classifier.
5. Evaluation: Assess model performance using classification metrics.
6. Visualization: Generate sentiment distribution charts, word clouds, and confusion matrix heatmaps.
7. Prediction: Classify new text into sentiment categories.

Dataset

The project uses the Sentiment140 dataset, which contains tweets and sentiment labels for sentiment analysis.

Model Evaluation

The Logistic Regression model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Visualizations

- Sentiment distribution charts
- Word clouds
- Confusion matrix heatmaps
- Sentiment-based visualization for user-entered text

Getting Started

Clone the repository:

git clone <your-repository-url>
cd SocialSway-Sentiment-Analysis

Install the required dependencies:

pip install -r requirements.txt

Run the project's main Python script or notebook according to the files available in the repository.

My Role

Led the SocialSway project as part of the Machine Learning Techniques course, focusing on the development and implementation of a machine learning-based sentiment analysis pipeline.

Conclusion

SocialSway demonstrates how machine learning and NLP techniques can be used to analyze sentiment in social media text. By combining text preprocessing, TF-IDF feature extraction, Logistic Regression, model evaluation, and visualization, the project provides a practical application of sentiment analysis.

---
