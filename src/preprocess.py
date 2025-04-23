import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib

def load_data(path):
    df = pd.read_csv(path, encoding="latin-1", header=None)
    df.columns = ['target','id','date','flag','user','text']
    df['sentiment'] = df['target'].apply(lambda x: 1 if x==4 else 0)
    return df[['text','sentiment']]

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|[^a-zA-Z\s]","", text)
    return text

def main():
    # 1) Carga y limpieza
    df = load_data("data/raw/sentiment140.csv")
    df['text_clean'] = df['text'].apply(clean_text)

    # 2) Vectorización TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['text_clean'])
    y = df['sentiment']

    # 3) División train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Asegúrate de que existan las carpetas
    import os
    os.makedirs("models", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    # 4) Guarda objetos
    joblib.dump(vectorizer, "models/tfidf_vectorizer.joblib")
    pd.DataFrame({'sentiment': y_train}).to_csv("data/processed/y_train.csv", index=False)
    joblib.dump(X_train, "data/processed/X_train.joblib")
    joblib.dump(X_test,  "data/processed/X_test.joblib")
    joblib.dump(y_test,  "data/processed/y_test.joblib")

    print("Preprocesamiento y vectorización listos.")

if __name__=="__main__":
    main()
