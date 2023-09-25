import joblib
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pickle
import matplotlib.pyplot as plt

def load_sentiment_model(model_path):

    model = joblib.load(model_path)
    return model


def text_process(message):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Convert all text to lowercase
    4. Returns a list of the cleaned text
    5. Remove short words (e.g., words with 2 or fewer characters)
    6. Returns a list of the cleaned text
    """

    STOPWORDS = set(stopwords.words('english')).union(
        ['u', 'ü', 'ur', '4', '2', 'im', 'dont', 'doin', 'ure'])

    # Tokenize the text
    words = word_tokenize(message)

    # Check characters to see if they are in punctuation
    nopunc = [word for word in words if word not in string.punctuation]

    # Convert the text to lowercase and remove short words
    cleaned_words = [word.lower() for word in nopunc if len(word) > 2]

    # Now just remove any stopwords
    return ' '.join([word for word in cleaned_words if word not in STOPWORDS])


def predict_sentiment(text, model_path='./static/models/my_best_sentiment_model_1695566256.8833115.sav'):

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

    correct_count = 0
    incorrect_count = 0

    for input_text, actual_sentiment in zip(input_texts, actual_sentiments):
        predicted_sentiment = predict_sentiment(input_text)

        if predicted_sentiment == actual_sentiment:
            correct_count += 1
        else:
            incorrect_count += 1

        print(f"Input Text: {input_text}")
        print(f"Predicted Sentiment: {predicted_sentiment}")
        print(f"Actual Sentiment: {actual_sentiment}")
        print("\n")

    # Create a bar chart to visualize the results
    labels = ['Correct', 'Incorrect']
    counts = [correct_count, incorrect_count]

    plt.bar(labels, counts, color=['green', 'red'])
    plt.xlabel('Predictions')
    plt.ylabel('Count')
    plt.title('Sentiment Analysis Results')
    plt.show()
