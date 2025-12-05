from pathlib import Path
import joblib

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

project_dir = Path.cwd()
csv_dir = project_dir / "data/trajectory_features.csv"

df = pd.read_csv(csv_dir, sep=";")

feature_cols = ["min_odleglosc_pix", "predkosc", "kat", "czy_w_tunelu_pod_obrecza", "net_moved"]

X = df[feature_cols]
y = df["scored"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

knn_pipeline = Pipeline([
    ("scaler", MinMaxScaler()),
    ("knn", KNeighborsClassifier(
        n_neighbors=5,
    ))
])

knn_pipeline.fit(X_train, y_train)

y_pred = knn_pipeline.predict(X_test)

print("Dokładność (accuracy):", accuracy_score(y_test, y_pred))
print("Macierz pomyłek:")
print(confusion_matrix(y_test, y_pred))
print("Raport klasyfikacji:")
print(classification_report(y_test, y_pred))

joblib.dump(knn_pipeline, "knn_model.pkl")
