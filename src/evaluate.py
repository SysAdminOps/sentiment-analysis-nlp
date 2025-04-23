import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

def main():
    # Carga
    X_test = joblib.load("data/processed/X_test.joblib")
    y_test = joblib.load("data/processed/y_test.joblib")
    model = joblib.load("models/logistic_regression.joblib")

    # Predicción
    y_pred = model.predict(X_test)

    # Matriz y display
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["neg","pos"])
    disp.plot()            # usa matplotlib internamente
    plt.title("Matriz de Confusión")
    plt.show()

if __name__=="__main__":
    main()
