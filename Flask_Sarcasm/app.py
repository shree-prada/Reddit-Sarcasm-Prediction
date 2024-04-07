import pickle
from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer  # Assuming TF-IDF


model = pickle.load(open('fore.pkl', 'rb'))
app = Flask(__name__, static_url_path='/static')


def predict_sarcasm(sentence_input):
  predicted_sarcasm = model.predict(sentence_input)
  return predicted_sarcasm

@app.route('/')
def first():
  return render_template("index.html")

@app.route('/result')
def index():
  return render_template('results.html')

@app.route('/predict', methods=['POST'])
def predict():
  # Get user input
  sentence_input = request.form.get('input')

  # Validation: Check if input is a string
  if not isinstance(sentence_input, str):
    return render_template('results.html', error_message="Invalid input. Please enter a text string.")

  # Get the sarcasm prediction
  predicted_sarcasm = predict_sarcasm(sentence_input)

  return render_template('results.html', prediction=predicted_sarcasm)

if __name__ == '__main__':
  app.run(debug=True)
