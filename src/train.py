import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

def main():
    # — Carga de datos —
    X_train = joblib.load("data/processed/X_train.joblib")
    y_train = pd.read_csv("data/processed/y_train.csv")["sentiment"]
    X_test  = joblib.load("data/processed/X_test.joblib")
    y_test  = joblib.load("data/processed/y_test.joblib")

    # — Entrenamiento —
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # — Evaluación —
    y_pred = model.predict(X_test)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

    # — Guardar modelo —
    joblib.dump(model, "models/logistic_regression.joblib")
    print("\nModelo guardado en models/logistic_regression.joblib")

if __name__ == "__main__":
    main()
