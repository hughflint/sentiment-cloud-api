# Let's upload all relevant libraries

from flask import Flask, request, jsonify, send_file
import tensorflow as tf
import numpy as np
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('snowball_data')
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
import re

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle

# Set here the maximum length sequence used during training
MAX_SEQUENCE_LENGTH = 140

# Let's also load stopwords used for preprossings
stop_words = stopwords.words('english')

# Let's set the cleaning pattern
text_cleaning_re = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

# We also need the tokenizer used for the training. It had been saved with pickle
# Let's load it using pickle
with open("./tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)

# Finally, let's load the Keras LSTM model trained
classification_model = load_model("./model.h5")
print(classification_model.summary())

def preprocess(text, stem_or_lem="lem"):
  text = re.sub(text_cleaning_re, ' ', str(text).lower()).strip()
  tokens = []
  for token in text.split():
    if token not in stop_words:
      if stem_or_lem == "stem":
        stemmer = SnowballStemmer('english')
        tokens.append(stemmer.stem(token))
      else:
        lemmatizer = WordNetLemmatizer()
        tokens.append(lemmatizer.lemmatize(token))
  return " ".join(tokens)


def predict_sentiment(text):
    # First let's preprocess the text in the same way than for the training
    text = preprocess(text)

    # Let's get the index sequences from the tokenizer
    index_sequence = pad_sequences(tokenizer.texts_to_sequences([text]),
                                   maxlen=MAX_SEQUENCE_LENGTH)

    probability_score = classification_model.predict(index_sequence)[0][0]

    if probability_score < 0.5:
        sentiment = "negative"
    else:
        sentiment = "positive"

    return sentiment, probability_score

app = Flask(__name__)

# This is the route to the API
@app.route("/predict_sentiment", methods=["POST"])
def predict():

    # Get the text included in the request
    text = request.args['text']

    # Process the text in order to get the sentiment
    results = predict_sentiment(text)

    return jsonify(text=text, sentiment=results[0], probability=str(results[1]))

# This is the reoute to the welcome page
@app.route("/")
def home():
    return "Hello, welcome on the sentiment classification API !"