from pathlib import Path
import joblib

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

project_dir = Path.cwd()
csv_dir = project_dir / "data/trajectory_features.csv"

df = pd.read_csv(csv_dir)

##feature_cols = ["min_odleglosc_pix", "predkosc", "kat", "czy_w_tunelu_pod_obrecza", "net_moved"]
feature_cols = ["min_odleglosc_pix", "czy_w_tunelu_pod_obrecza"]

X = df[feature_cols]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)


def train_knn():
    knn_pipeline = Pipeline([
        ("scaler", MinMaxScaler()),
        ("knn", KNeighborsClassifier(
            n_neighbors=7,
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
    
def train_dt():
    dt_pipeline = Pipeline([
        ("dt", DecisionTreeClassifier(
            random_state=42,
            max_depth=5,
            min_samples_split=10
        ))
    ])

    dt_pipeline.fit(X_train, y_train)

    y_pred = dt_pipeline.predict(X_test)

    print("Dokładność (accuracy):", accuracy_score(y_test, y_pred))
    print("Macierz pomyłek:")
    print(confusion_matrix(y_test, y_pred))
    print("Raport klasyfikacji:")
    print(classification_report(y_test, y_pred))

    joblib.dump(dt_pipeline, "dt_model.pkl")
    
def train_lt():
    lt_pipeline = Pipeline([
        ("tl", LogisticRegression(random_state=42))
    ])

    lt_pipeline.fit(X_train, y_train)

    y_pred = lt_pipeline.predict(X_test)

    print("Dokładność (accuracy):", accuracy_score(y_test, y_pred))
    print("Macierz pomyłek:")
    print(confusion_matrix(y_test, y_pred))
    print("Raport klasyfikacji:")
    print(classification_report(y_test, y_pred))

    joblib.dump(lt_pipeline, "lt_model.pkl")
    
train_knn()