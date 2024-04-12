import pickle

import string

import nltk

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer

import spacy

import numpy as np
 
from flask import Flask, render_template, request, url_for, jsonify
 
# Initialize SpaCy

nlp = spacy.load("en_core_web_sm")
 
# Initialize the NLTK tokenizer, lemmatizer, and stopwords

nltk.download('punkt')

nltk.download('stopwords')

nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

stop_words = set(stopwords.words('english'))
 
# Initialize the Tf-Idf vectorizer

vectorizer = TfidfVectorizer()
 
# Function to preprocess text

def preprocess_text(text):

    doc = nlp(text)

    preprocessed_tokens = []

    for token in doc:

        # Lowercase the token if it's not in all caps

        if not token.text.isupper():

            token_text = token.text.lower()

        else:

            token_text = token.text

        # Lemmatize the token

        preprocessed_tokens.append(token_text)

    return ' '.join(preprocessed_tokens)
 
# Function to compute average GloVe embeddings for a text

def compute_average_glove_embedding(text, nlp_model):

    doc = nlp_model(text)

    # Get vectors for each token in the text

    word_vectors = [token.vector for token in doc if not token.is_stop]

    if word_vectors:

        # Compute average embedding

        average_embedding = np.mean(word_vectors, axis=0)

        return average_embedding

    else:

        # Return zeros if no valid word vectors found

        return np.zeros(nlp_model.vocab.vectors_length)
 
# Function to predict sarcasm

def predict_sarcasm(input_data):

    # Preprocess input data

    preprocessed_data = [preprocess_text(sentence) for sentence in input_data]

    # Vectorize preprocessed data

    vectorized_data = vectorizer.transform(preprocessed_data)

    # Predict sarcasm

    predicted_sarcasm = model.predict(vectorized_data)

    return predicted_sarcasm
 
app = Flask(__name__, static_url_path='/static')

model = pickle.load(open('fore.pkl', 'rb'))
 
@app.route('/')

def first():

    return render_template("first.html")
 
@app.route('/index')

def index():

    return render_template('index.html')
 
@app.route('/predict', methods=['POST'])

def predict():

    input_sentence = request.form['input_sentence']

    input_data = [input_sentence]
 
    # Compute average GloVe embeddings

    glove_embedding = compute_average_glove_embedding(input_sentence, nlp)

    # Preprocess the text after GloVe embedding

    preprocessed_text = preprocess_text(input_sentence)

    # Compute TF-IDF

    tfidf_matrix = vectorizer.fit_transform([preprocessed_text])

    # Convert TF-IDF matrix to array

    tfidf_array = tfidf_matrix.toarray()

    # Combine GloVe embedding with TF-IDF array

    combined_matrix = np.concatenate((glove_embedding.reshape(1, -1), tfidf_array), axis=1)
 
    # Predict sarcasm

    predicted_sarcasm = predict_sarcasm(input_data)
 
    return render_template('index.html', prediction=predicted_sarcasm)
 
if __name__ == '__main__':

    app.run(debug=True)
