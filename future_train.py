import requests
import pandas as pd
import pickle
import os
from retrain import preprocess
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier

# You can get a free API key from Google Cloud Console
# For now, I'm providing a structure that works if you add the key
GOOGLE_API_KEY = 'YOUR_GOOGLE_FACT_CHECK_API_KEY'

def fetch_fact_checks(query='world news'):
    """Fetches latest fact-checked claims from Google API"""
    print(f"Fetching latest fact-checks from Google API...")
    url = f"https://factchecktools.googleapis.com/v1alpha1/claims:search?query={query}&key={GOOGLE_API_KEY}"
    
    try:
        response = requests.get(url)
        data = response.json()
        
        claims_list = []
        if 'claims' in data:
            for claim in data['claims']:
                text = claim.get('text', '')
                claim_reviews = claim.get('claimReview', [])
                if claim_reviews:
                    rating = claim_reviews[0].get('textualRating', '').lower()
                    # Mapping ratings to Label
                    if any(fake_word in rating for fake_word in ['false', 'incorrect', 'fake', 'manipulated']):
                        label = 'FAKE'
                    elif any(real_word in rating for real_word in ['true', 'correct', 'real']):
                        label = 'REAL'
                    else:
                        continue # Skip ambiguous ones
                    
                    claims_list.append({'text': text, 'label': label})
        
        return pd.DataFrame(claims_list)
    except Exception as e:
        print(f"Error fetching from Google API: {e}")
        return pd.DataFrame()

def train_for_future():
    # 1. Load Past Data
    print("Step 1: Loading Past Data...")
    df_past = pd.read_csv('dataset/train.csv')[['text', 'label']]
    
    # 2. Fetch Latest/Current Fake & Real news from API
    print("Step 2: Fetching Current/Future Data from Google Fact Check API...")
    df_current = fetch_fact_checks('politics') # Change query as needed
    
    if not df_current.empty:
        # Merge Past and Current
        df_combined = pd.concat([df_past, df_current], ignore_index=True)
        print(f"New Data Added: {len(df_current)} rows. Total: {len(df_combined)}")
    else:
        df_combined = df_past
        print("Using past data only (Check API Key).")

    # 3. Preprocess for better Future Prediction
    print("Step 3: Preprocessing...")
    df_combined = df_combined.dropna()
    df_combined['text'] = df_combined['text'].apply(preprocess)
    
    # 4. Final Training
    print("Step 4: Training for Past, Current, and Future...")
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    X = vectorizer.fit_transform(df_combined['text'])
    y = df_combined['label']
    
    pac = PassiveAggressiveClassifier(max_iter=50)
    pac.fit(X, y)
    
    # 5. Saving the future-ready model
    pickle.dump(pac, open("model.pkl", "wb"))
    pickle.dump(vectorizer, open("vector.pkl", "wb"))
    print("MISSION COMPLETE! The model is now trained to handle future trends.")

if __name__ == "__main__":
    if GOOGLE_API_KEY == 'YOUR_GOOGLE_FACT_CHECK_API_KEY':
        print("\n!!! NOTE: Add your Google API Key in 'future_train.py' to fetch LIVE data !!!")
        print("Continuing with local simulation...")
        train_for_future()
    else:
        train_for_future()
