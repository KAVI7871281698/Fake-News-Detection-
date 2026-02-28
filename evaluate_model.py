import pandas as pd
import pickle
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score, classification_report

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Load the current model and vectorizer
try:
    loaded_model = pickle.load(open("model.pkl", 'rb'))
    vector = pickle.load(open("vector.pkl", 'rb'))
    print("Model and Vectorizer loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

lemmatizer = WordNetLemmatizer()
stpwrds = set(stopwords.words('english'))

def preprocess(news):
    if not isinstance(news, str):
        return ""
    review = re.sub(r'[^a-zA-Z\s]', '', news)
    review = review.lower()
    review = nltk.word_tokenize(review)
    corpus = [lemmatizer.lemmatize(y) for y in review if y not in stpwrds]
    return ' '.join(corpus)

# Load evaluation data (using a subset of train for speed or test if available)
print("Loading dataset...")
df = pd.read_csv('dataset/train.csv').head(1000) # Check first 1000 rows
df = df.dropna(subset=['text', 'label'])

print("Preprocessing evaluation data...")
df['clean_text'] = df['text'].apply(preprocess)

print("Vectorizing...")
X = vector.transform(df['clean_text'])
y = df['label']

print("Predicting...")
y_pred = loaded_model.predict(X)

print("\nAccuracy Score:", accuracy_score(y, y_pred))
print("\nClassification Report:\n", classification_report(y, y_pred))
