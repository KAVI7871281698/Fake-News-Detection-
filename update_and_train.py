import pandas as pd
from newsapi import NewsApiClient
import os
import pickle
from retrain import preprocess
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier

# Note: You need a free API key from https://newsapi.org/
NEWS_API_KEY = 'YOUR_NEWS_API_KEY_HERE' # Replace with actual key

def fetch_latest_news(query='world'):
    print(f"Fetching latest news for: {query}")
    try:
        newsapi = NewsApiClient(api_key=NEWS_API_KEY)
        # Fetching top headlines or everything
        all_articles = newsapi.get_everything(q=query, language='en', sort_by='relevancy', page_size=100)
        
        articles = all_articles['articles']
        new_data = []
        for art in articles:
            if art['description'] and art['content']:
                # For training purpose, we might need labels. 
                # Since we are fetching from news sources directly, we can assume them as 'REAL' 
                # or manually label them later. For a demo, let's mark them as REAL (1).
                new_data.append({
                    'text': str(art['title']) + " " + str(art['description']),
                    'label': 'REAL' 
                })
        
        return pd.DataFrame(new_data)
    except Exception as e:
        print(f"API Error: {e}")
        return pd.DataFrame()

def extended_train():
    # 1. Load Past Data
    print("Loading Past Data...")
    df_past = pd.read_csv('dataset/train.csv')[['text', 'label']]
    
    # 2. Fetch Current/Live Data (Simulated or Real)
    print("Fetching Current Data via API...")
    df_current = fetch_latest_news('politics') # You can change the query
    
    if not df_current.empty:
        # Merge Past and Current
        df_combined = pd.concat([df_past, df_current], ignore_index=True)
        print(f"Total dataset size: {len(df_combined)}")
    else:
        df_combined = df_past
        print("Using past data only (API fetch failed or no key).")

    # 3. Preprocess
    print("Preprocessing...")
    df_combined = df_combined.dropna()
    df_combined['text'] = df_combined['text'].apply(preprocess)
    
    # 4. Train to handle Current and Future trends
    print("Training Model...")
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    X = vectorizer.fit_transform(df_combined['text'])
    y = df_combined['label']
    
    pac = PassiveAggressiveClassifier(max_iter=50)
    pac.fit(X, y)
    
    # 5. Save for Future Predictions
    pickle.dump(pac, open("model.pkl", "wb"))
    pickle.dump(vectorizer, open("vector.pkl", "wb"))
    print("Success! Model is now ready for Past, Current, and Future predictions.")

if __name__ == "__main__":
    # If the user hasn't provided API key, this will only show the logic.
    if NEWS_API_KEY == 'YOUR_NEWS_API_KEY_HERE':
        print("PLEASE ADD YOUR NEWSAPI KEY FIRST!")
    else:
        extended_train()
