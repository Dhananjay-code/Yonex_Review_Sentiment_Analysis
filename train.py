import pandas as pd
import re
import joblib
import nltk
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Setup NLTK
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))

# Set the Experiment Name for MLflow
mlflow.set_experiment("Yonex_Sentiment_Analysis_Experiment")

def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.replace("READ MORE", "")
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words])

# Start MLflow Tracking
with mlflow.start_run(run_name="Logistic_Regression_Training"):
    
    # 1. Load Data (Updated path for Ubuntu)
    df = pd.read_csv("data.csv") 

    # 2. Preprocess
    df['sentiment'] = df['Ratings'].apply(lambda x: 1 if x > 3 else 0)
    df['cleaned_text'] = df['Review text'].apply(clean_text)

    # 3. Vectorization
    max_feats = 2500
    tfidf = TfidfVectorizer(max_features=max_feats)
    X = tfidf.fit_transform(df['cleaned_text'])
    y = df['sentiment']

    # 4. Training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # 5. Evaluate Metrics
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)

    # --- MLFLOW LOGGING ---
    # Log Hyperparameters
    mlflow.log_param("max_features", max_feats)
    mlflow.log_param("model_type", "LogisticRegression")
    
    # Log Metrics (As required by your task)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)

    # Log the model (Artifact & Registry)
    mlflow.sklearn.log_model(model, "sentiment_model")
    
    # 6. Save Local Files for the Website
    joblib.dump(model, 'sentiment_model.pkl')
    joblib.dump(tfidf, 'tfidf_vectorizer.pkl')

    print(f"Success! Model Accuracy: {acc:.4f}")
    print("Run tracked in MLflow. Metrics and Model are now saved.")
