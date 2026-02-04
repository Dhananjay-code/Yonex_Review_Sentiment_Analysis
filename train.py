import pandas as pd
import re
import joblib
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Setup NLTK
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    if not isinstance(text, str): return ""
    # Remove "READ MORE" which appears in Flipkart samples
    text = text.replace("READ MORE", "")
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words])

# 1. Load Data
df = pd.read_csv("D:\\DJ\\Innomatics Internship\\reviews_data_dump\\reviews_badminton\\data.csv") # Ensure your file is named this

# 2. Preprocess
# Labeling: Ratings 4 & 5 are Positive (1), 1-3 are Negative (0)
df['sentiment'] = df['Ratings'].apply(lambda x: 1 if x > 3 else 0)
df['cleaned_text'] = df['Review text'].apply(clean_text)

# 3. Vectorization
tfidf = TfidfVectorizer(max_features=2500)
X = tfidf.fit_transform(df['cleaned_text'])
y = df['sentiment']

# 4. Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# 5. Save Model and Vectorizer
joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
print("Model trained and saved successfully!")