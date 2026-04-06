import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import pickle

# 1. Load Dataset
try:
    # Ensure this filename matches your CSV exactly!
    df = pd.read_csv('cyberbullying_tweets.csv', encoding='latin1', on_bad_lines='skip')
    
    # Map labels: 'not_cyberbullying' -> 0, everything else -> 1
    df['label'] = df['cyberbullying_type'].apply(lambda x: 0 if x == 'not_cyberbullying' else 1)
    
    # 2. Pipeline: NLP (TF-IDF) + ML (Logistic Regression)
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
        ('lr', LogisticRegression())
    ])

    # 3. Train
    print("Training started... please wait.")
    X_train, X_test, y_train, y_test = train_test_split(df['tweet_text'], df['label'], test_size=0.2)
    pipeline.fit(X_train, y_train)

    # 4. Save the Brain
    with open('model.pkl', 'wb') as f:
        pickle.dump(pipeline, f)
    print("✅ Success: model.pkl created!")

except Exception as e:
    print(f"❌ Error: {e}")