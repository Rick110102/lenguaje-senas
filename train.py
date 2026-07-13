"""
Etapa 4: Entrenamiento del modelo.
Carga el dataset de landmarks, entrena un Random Forest, evalua
su desempeno y guarda el modelo entrenado en disco.
"""

import os
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

DATASET_PATH = "dataset.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "sign_classifier.pkl")


def main():
    df = pd.read_csv(DATASET_PATH)
    print(f"Dataset cargado: {len(df)} muestras, {df['label'].nunique()} letras distintas")
    print(df["label"].value_counts())

    X = df.drop(columns=["label"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nEntrenamiento: {len(X_train)} muestras | Prueba: {len(X_test)} muestras")

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nAccuracy en el set de prueba: {accuracy:.2%}")
    print("\nReporte por letra:")
    print(classification_report(y_test, y_pred))

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    print(f"Modelo guardado en {MODEL_PATH}")


if __name__ == "__main__":
    main()