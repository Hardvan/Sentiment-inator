from flask import Flask, render_template, request, redirect, url_for


# Import the predict_sentiment function
import NLP_model

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():

    result = None
    if request.method == 'POST':
        result = {}

        # Text input from the form
        text_input = request.form['text_input']
        result["text_input"] = text_input

        # Predict the sentiment
        predicted_sentiment = None  # positive, neutral, negative
        if text_input:
            predicted_sentiment, result_color = NLP_model.predict_sentiment(
                text_input)
        else:
            predicted_sentiment = "Please enter some text."
            result_color = "blue"

        result["predicted_sentiment"] = predicted_sentiment
        result["result_color"] = result_color

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)
