from flask import Flask, render_template, request
import joblib
import re
from nltk.stem import WordNetLemmatizer
import nltk
import numpy as np

# Initialize Flask and load model
app = Flask(__name__)
model = joblib.load('sentiment_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')
lemmatizer = WordNetLemmatizer()

def predict_sentiment(text):
    # Minimal cleaning for real-time input
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    cleaned = " ".join([lemmatizer.lemmatize(w) for w in text.split()])
    vectorized = tfidf.transform([cleaned])
    prediction = model.predict(vectorized)
    return "Positive" if prediction[0] == 1 else "Negative"

@app.route('/', methods=['GET', 'POST'])
def home():
    sentiment = None
    confidence = None
    
    if request.method == 'POST':
        review = request.form.get('review')
        if review:
            # 1. Vectorize input
            vectorized = tfidf.transform([review])
            
            # 2. Get Prediction and Probability
            prediction = model.predict(vectorized)[0]
            proba = model.predict_proba(vectorized)[0] # Returns [neg_prob, pos_prob]
            
            sentiment = "Positive" if prediction == 1 else "Negative"
            # Get the confidence of the chosen class
            confidence = round(np.max(proba) * 100, 2)
            
    return render_template('index.html', sentiment=sentiment, confidence=confidence)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)