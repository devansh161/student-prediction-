import argparse
import json
import os
import glob
import random

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

FEATURES = [
    "G1", "G2", "studytime", "failures", "absences", "age",
    "Medu", "Fedu", "goout", "Dalc", "health", "higher",
    "schoolsup", "freetime"
]
CLASS_NAMES = ["Low", "Medium", "High"]

def find_dataset(requested_path=None):
    candidates = []
    if requested_path:
        candidates.append(requested_path)
    candidates += [
        "student-por.csv",
        "/kaggle/input/**/student-por.csv",
    ]
    for candidate in candidates:
        matches = glob.glob(candidate, recursive=True)
        for match in matches:
            if os.path.isfile(match):
                return match
    raise FileNotFoundError(
        "Could not find student-por.csv. Download the UCI Portuguese dataset "
        "and place student-por.csv beside this script, or pass --data /path/to/student-por.csv."
    )

def grade_category(grade):
    if grade <= 9:
        return 0
    if grade <= 14:
        return 1
    return 2

def build_model(input_dim):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(3, activation="softmax"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None, help="Path to student-por.csv")
    parser.add_argument("--output", default="artifacts", help="Folder for saved model files")
    parser.add_argument("--epochs", type=int, default=150)
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(42)
    data_path = find_dataset(args.data)
    print(f"Loading: {data_path}")

    df = pd.read_csv(data_path, sep=";")
    required = FEATURES + ["G3"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    df = df.copy()
    df["target"] = df["G3"].apply(grade_category)
    df["higher"] = (df["higher"].astype(str).str.lower() == "yes").astype(int)
    df["schoolsup"] = (df["schoolsup"].astype(str).str.lower() == "yes").astype(int)

    X = df[FEATURES].apply(pd.to_numeric, errors="raise")
    y = df["target"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = build_model(len(FEATURES))
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True
    )
    history = model.fit(
        X_train_scaled, y_train,
        validation_split=0.2,
        epochs=args.epochs,
        batch_size=16,
        callbacks=[early_stop],
        verbose=1,
    )

    y_pred = np.argmax(model.predict(X_test_scaled, verbose=0), axis=1)
    dnn_accuracy = accuracy_score(y_test, y_pred)
    print(f"\nDNN test accuracy: {dnn_accuracy:.4f}")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_scaled, y_train)

    comparison = {
        "DNN": float(dnn_accuracy),
        "Random Forest": float(accuracy_score(y_test, rf.predict(X_test))),
        "Logistic Regression": float(accuracy_score(y_test, lr.predict(X_test_scaled))),
    }
    print("\nModel comparison:")
    for name, score in comparison.items():
        print(f"{name}: {score:.4f}")

    os.makedirs(args.output, exist_ok=True)
    model.save(os.path.join(args.output, "model.keras"))
    joblib.dump(scaler, os.path.join(args.output, "scaler.pkl"))
    joblib.dump(FEATURES, os.path.join(args.output, "features.pkl"))
    with open(os.path.join(args.output, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump({
            "class_names": CLASS_NAMES,
            "features": FEATURES,
            "dnn_test_accuracy": dnn_accuracy,
            "model_comparison": comparison,
            "training_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "epochs_completed": len(history.history["loss"]),
            "target_rule": "Low: G3 <= 9; Medium: 10-14; High: 15-20",
        }, f, indent=2)

    print(f"\nSaved model, scaler, features and metadata in: {args.output}")

if __name__ == "__main__":
    main()
