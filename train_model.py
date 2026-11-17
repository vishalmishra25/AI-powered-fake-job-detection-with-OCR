from pathlib import Path
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "sample_jobs.csv"
MODEL_PATH = BASE_DIR / "models" / "fake_job_model.pkl"


def train_model():
    df = pd.read_csv(DATA_PATH).fillna("")
    required = {"title", "company_profile", "description", "requirements", "benefits", "fraudulent"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    text_columns = ["title", "company_profile", "description", "requirements", "benefits"]
    X = df[text_columns].astype(str).agg(" ".join, axis=1)
    y = df["fraudulent"].astype(int)
    if y.nunique() < 2:
        raise ValueError("Dataset must contain both genuine (0) and fraudulent (1) examples.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=12000, min_df=1)),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("\n=== Baseline evaluation on held-out demo rows ===")
    print(f"Accuracy : {accuracy_score(y_test, predictions):.3f}")
    print(f"Precision: {precision_score(y_test, predictions, zero_division=0):.3f}")
    print(f"Recall   : {recall_score(y_test, predictions, zero_division=0):.3f}")
    print(f"F1-score : {f1_score(y_test, predictions, zero_division=0):.3f}")
    print(classification_report(y_test, predictions, target_names=["Genuine (0)", "Fraudulent (1)"], zero_division=0))

    MODEL_PATH.parent.mkdir(exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"\nSaved model to: {MODEL_PATH}")
    print("IMPORTANT: these metrics are only for the small bundled demonstration dataset, not real-world performance.")
    return model


if __name__ == "__main__":
    train_model()
