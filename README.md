# Text-Mining-Project
Sentiment Analysis Showdown: VADER vs. RoBERTa on Amazon Reviews
Sentiment Analysis Showdown: VADER vs. RoBERTa on Amazon Reviews

🧐 About The Project

This project explores and compares different methods for performing sentiment analysis on the Amazon Fine Food Reviews dataset. The goal is to evaluate the effectiveness of a classic rule-based model against a modern transformer-based model by comparing their sentiment predictions to the actual star ratings provided by users.

The notebook processes a sample of 500 reviews and applies the following methodologies:

    NLTK VADER: A lexicon and rule-based sentiment analysis tool.

    RoBERTa: A pre-trained transformer model from Hugging Face fine-tuned for sentiment analysis.

    Hugging Face Pipeline: A high-level interface for quick sentiment prediction using a default model.

🛠️ Methodologies Explored

    NLTK VADER (Valence Aware Dictionary and Sentiment Reasoner)

    This is a "bag of words" approach that scores text based on a dictionary of words with pre-assigned sentiment scores. It's fast and doesn't require training but lacks a deep understanding of context.

    🤖 RoBERTa (from Hugging Face)

    This project uses the cardiffnlp/twitter-roberta-base-sentiment model, a powerful transformer-based model. Unlike VADER, RoBERTa understands the context of words in a sentence, leading to more nuanced and accurate sentiment predictions.

    🤗 Hugging Face Pipeline

    A quick demonstration of the high-level sentiment-analysis pipeline from the Transformers library, which provides an easy way to get predictions from a pre-trained model (in this case, distilbert-base-uncased-finetuned-sst-2-english).

📊 Project Steps

    Data Loading and EDA: The Amazon Fine Food Reviews dataset is loaded, and a sample of 500 reviews is taken for analysis. A quick exploratory data analysis (EDA) is performed to visualize the distribution of star ratings.

    NLTK Basics: The notebook demonstrates basic Natural Language Processing (NLP) tasks using NLTK, including tokenization, part-of-speech (POS) tagging, and named entity recognition (NER).

    VADER Analysis: NLTK's SentimentIntensityAnalyzer is used to calculate negative, neutral, positive, and compound sentiment scores for each review. These scores are then visualized against the actual star ratings.

    RoBERTa Analysis: The pre-trained RoBERTa model is used to predict the probability of each review being negative, neutral, or positive.

    Comparison and Visualization: The results from both VADER and RoBERTa are combined into a single DataFrame. A pairplot is generated to visually compare the correlations between the models' scores and the ground-truth star ratings.

    Qualitative Review: The analysis concludes by examining specific examples where the models' predictions differ significantly from the user's star rating, highlighting the strengths and weaknesses of each approach.

🚀 How to Use

    Set up your environment: Ensure you have Python 3 installed.

    Install libraries:
    Bash

pip install pandas numpy matplotlib seaborn nltk torch transformers tqdm

Download NLTK data: Run the following commands in a Python interpreter to download the necessary NLTK packages.
Python

    import nltk
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('maxent_ne_chunker')
    nltk.download('words')
    nltk.download('vader_lexicon')

    Run the Notebook: Open the sentiment-analysis.ipynb file in a Jupyter environment and run the cells sequentially.

📚 Key Libraries Used

    Pandas

    NLTK

    Transformers (Hugging Face)

    PyTorch

    Matplotlib & Seaborn

