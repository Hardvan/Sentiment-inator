import joblib
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import re
import time

import nltk
nltk.download('stopwords')


def load_sentiment_model(model_path):

    model = joblib.load(model_path)
    return model


def text_process(message):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Remove links (HTTP, //, etc.)
    4. Convert all text to lowercase
    5. Remove short words (e.g., words with 2 or fewer characters)
    6. Returns a cleaned string
    """

    STOPWORDS = set(stopwords.words('english')).union(
        ['u', 'ü', 'ur', '4', '2', 'im', 'dont', 'doin', 'ure'])

    # Remove links using regular expressions
    message = re.sub(r'http\S+', '', message)  # Remove HTTP links
    message = re.sub(r'www\.\S+', '', message)   # Remove www links
    message = re.sub(r'\/\/\S+', '', message)    # Remove // links
    message = re.sub('\[.*?\]', '', message)
    message = re.sub('https?://\S+|www\.\S+', '', message)
    message = re.sub('<.*?>+', '', message)
    message = re.sub('[%s]' % re.escape(string.punctuation), '', message)
    message = re.sub('\n', '', message)
    message = re.sub('\w*\d\w*', '', message)

    # Tokenize the text
    words = word_tokenize(message)

    # Check characters to see if they are in punctuation
    nopunc = [word for word in words if word not in string.punctuation]

    # Convert the text to lowercase and remove short words
    cleaned_words = [word.lower() for word in nopunc if len(word) > 2]

    # Now just remove any stopwords and return a cleaned string
    return ' '.join([word for word in cleaned_words if word not in STOPWORDS])


def predict_sentiment(text, model_path='./static/models/my_best_sentiment_model_1695647318.9605517.sav'):

    model = load_sentiment_model(model_path)

    # Perform the necessary text preprocessing
    cleaned_text = text_process(text)

    # Vectorize the cleaned text
    vect = pickle.load(open("./static/models/my_vect.pkl", "rb"))
    text_dtm = vect.transform([cleaned_text])

    # Predict sentiment using the loaded model
    sentiment_label = model.predict(text_dtm)[0]

    # Map the numerical sentiment label back to the original sentiment
    sentiment_mapping = {0: 'positive', 1: 'neutral', 2: 'negative'}
    predicted_sentiment = sentiment_mapping[sentiment_label]

    return predicted_sentiment


if __name__ == '__main__':

    mode = 2  # 1: small test, 2: large test

    if mode == 1:
        input_texts = [
            "I love this product, it's amazing!",
            "This is a neutral statement.",
            "I dislike this product, it's terrible!",
            "Great experience with the service!",
            "The movie was fantastic and thrilling.",
            "It's just okay, not too bad.",
            "This book is a masterpiece.",
            "The weather today is terrible.",
            "The event was so-so, not very exciting.",
            "I'm feeling great today."
        ]
        actual_sentiments = ['positive', 'neutral', 'negative', 'positive', 'positive',
                             'neutral', 'positive', 'negative', 'neutral', 'positive']
    elif mode == 2:
        test = pd.read_csv('./static/data/test.csv')

        input_texts = test['text'].values
        actual_sentiments = test['sentiment'].values

    correct_count = 0
    incorrect_count = 0

    # Initialize variables to store true and predicted labels
    true_labels = []
    predicted_labels = []

    start_time = time.time()

    for input_text, actual_sentiment in zip(input_texts, actual_sentiments):
        predicted_sentiment = predict_sentiment(input_text)

        true_labels.append(actual_sentiment)
        predicted_labels.append(predicted_sentiment)

        if predicted_sentiment == actual_sentiment:
            correct_count += 1
        else:
            incorrect_count += 1

        # print(f"Input Text: {input_text}")
        # print(f"Predicted Sentiment: {predicted_sentiment}")
        # print(f"Actual Sentiment: {actual_sentiment}")
        # print("\n")

    # Create a bar chart to visualize the results
    labels = ['Correct', 'Incorrect']
    counts = [correct_count, incorrect_count]

    plt.figure(figsize=(8, 6), dpi=200)
    plt.bar(labels, counts, color=['green', 'red'])
    plt.xlabel('Predictions')
    plt.ylabel('Count')
    plt.title('Sentiment Analysis Results')

    # Save the bar chart as an image under static/graphs folder
    plt.savefig('./static/graphs/sentiment_bar_chart.png')

    plt.show()

    # Calculate the confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels, labels=[
        'positive', 'neutral', 'negative'])

    print("Confusion Matrix:")
    print(cm)

    # Plot the confusion matrix as a heatmap
    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
    im = ax.imshow(cm, interpolation='nearest', cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=['positive', 'neutral', 'negative'],
           yticklabels=['positive', 'neutral', 'negative'],
           title='Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')

    # Loop over data dimensions and create text annotations
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()

    # Save the confusion matrix plot as an image under static/graphs folder
    plt.savefig('./static/graphs/confusion_matrix.png')

    plt.show()

    end_time = time.time()
    elapsed_time = round(end_time - start_time, 2)
    print(f"Elapsed time: {elapsed_time} seconds")

    # Accuracy
    accuracy = round(
        correct_count / (correct_count + incorrect_count) * 100, 2)
    print(f"Accuracy: {accuracy}%")
