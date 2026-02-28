import pandas as pd
import pickle
import os
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split

# Ensure NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()
stpwrds = set(stopwords.words('english'))

def preprocess(news):
    if not isinstance(news, str): return ""
    review = re.sub(r'[^a-zA-Z\s]', '', news)
    review = review.lower()
    review = nltk.word_tokenize(review)
    corpus = [lemmatizer.lemmatize(y) for y in review if y not in stpwrds]
    return ' '.join(corpus)

def retrain_model():
    print("Starting retraining process...")
    
    # 1. Load original data
    print("Loading original training data...")
    df_orig = pd.read_csv('dataset/train.csv')
    df_orig = df_orig[['text', 'label']].dropna()
    
    # 2. Load feedback data if exists
    if os.path.exists('feedback_data.csv'):
        print("Loading feedback data...")
        try:
            # Safe reading of feedback data
            df_feedback = pd.read_csv('feedback_data.csv', names=['text', 'label'], on_bad_lines='skip')
            df_orig = pd.concat([df_orig, df_feedback], ignore_index=True)
            print(f"Added {len(df_feedback)} new entries from feedback.")
        except Exception as e:
            print(f"Error loading feedback: {e}")
            
    # 3. Preprocess
    print("Preprocessing data (this might take a while)...")
    df_orig['text'] = df_orig['text'].apply(preprocess)
    
    # 4. Vectorize
    print("Vectorizing...")
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    X = vectorizer.fit_transform(df_orig['text'])
    y = df_orig['label']
    
    # 5. Train
    print("Training PassiveAggressiveClassifier...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pac = PassiveAggressiveClassifier(max_iter=50)
    pac.fit(X_train, y_train)
    
    # 6. Save
    print("Saving new model and vectorizer...")
    pickle.dump(pac, open("model.pkl", "wb"))
    pickle.dump(vectorizer, open("vector.pkl", "wb"))
    
    print("DONE! Model updated successfully.")

if __name__ == "__main__":
    retrain_model()
