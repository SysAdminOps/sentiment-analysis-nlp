# Streamlit Sentiment Analysis App
# File: app.py

import streamlit as st
import joblib
import numpy as np

# Cargar vectorizador y modelo entrenado
vectorizer = joblib.load("models/tfidf_vectorizer.joblib")
model = joblib.load("models/logistic_regression.joblib")

# Título de la aplicación
st.title("Clasificador de Sentimientos NLP 📊")
st.write("Ingresa un texto (tweet o reseña) y el modelo predirá si el sentimiento es positivo o negativo.")

# Entrada de texto del usuario
user_input = st.text_area("Escribe tu texto aquí:", height=150)

if st.button("Analizar Sentimiento"):
    if user_input:
        # Limpieza básica (mismas transformaciones que en preprocess)
        import re
        def clean_text(text):
            text = text.lower()
            text = re.sub(r"http\S+|[^a-zA-Z\s]", "", text)
            return text

        cleaned = clean_text(user_input)
        # Vectorizar
        vect = vectorizer.transform([cleaned])
        # Predecir
        pred = model.predict(vect)[0]
        proba = model.predict_proba(vect)[0]
        label = "Positivo 👍" if pred == 1 else "Negativo 👎"
        confidence = np.max(proba)

        # Mostrar resultado
        st.markdown(f"**Sentimiento:** {label}")
        st.markdown(f"**Confianza:** {confidence:.2%}")
    else:
        st.warning("Por favor ingresa un texto para analizar.")

