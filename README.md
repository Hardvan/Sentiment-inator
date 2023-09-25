# Sentimentinator

## Project Flowchart

![NLP Flowchart](static/images/NLP%20Model%20Flowchart.png)

## Graphs for [Test Data](./static/data/test.csv)

### Confusion Matrix

![Confusion Matrix](./static/graphs/confusion_matrix.png)

### Accuracy Graph

![Accuracy Graph](./static/graphs/sentiment_bar_chart.png)

## Tech Stack

- Python (3.10.4)
- Flask
- HTML
- CSS
- JavaScript
- Natural Language Processing (NLP) Stack:
  - NLTK (Natural Language Toolkit)
  - Scikit-learn
  - Matplotlib
  - Pandas
  - Numpy
  - Regular Expressions (Regex)

## Installation

1. Clone the repo

   ```bash
   git clone https://github.com/Hardvan/Sentimentinator.git
   ```

2. Navigate to the folder

   ```bash
   cd Sentimentinator
   ```

3. Create a virtual python environment by typing the following in the terminal

   ```bash
   python -m venv .venv
   ```

4. Activate the virtual environment

   Windows:

   ```bash
   .\.venv\Scripts\activate
   ```

   Linux:

   ```bash
   source .venv/bin/activate
   ```

5. Install dependencies by typing the following in the terminal

   ```bash
   pip install -r requirements.txt
   ```

6. Run the app

   ```bash
   python app.py
   ```

7. Click on the link in the terminal to open the website

   It will look something like this:

   ```bash
   Running on http://127.0.0.1:5000
   ```
