"""
Naive Bayes Spam-Classifier (scikit-learn)
- lädt den Kaggle Spam-Datensatz (CSV)
- Bag-of-Words Features (CountVectorizer)
- Train/Test Split (stratify)
- Multinomial Naive Bayes Training
- Evaluation (Accuracy, Confusion Matrix, Classification Report)
- wichtigste Begriffe pro Klasse (ham/spam)
"""

from __future__ import annotations

import argparse
import sys
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def load_dataset(path: str) -> pd.DataFrame:

    df = pd.read_csv(path)

    required = {"text", "label_num"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"CSV muss die Spalten {required} enthalten. Gefunden: {set(df.columns)}"
        )

    df = df[["text", "label_num"]].dropna()
    # Sicherstellen int labels 0/1
    df["label_num"] = df["label_num"].astype(int)
    df = df[df["label_num"].isin([0, 1])]

    return df


def print_top_words_by_class(
    model: MultinomialNB,
    vectorizer: CountVectorizer,
    topk: int = 10,
) -> None:
    """
    Gibt Wörter mit höchster P(w|class) aus.
    """
    feature_names = vectorizer.get_feature_names_out()
    log_probs = model.feature_log_prob_  # shape: (n_classes, n_features)


    class_to_row = {c: i for i, c in enumerate(model.classes_)}

    ham_row = class_to_row.get(0)
    spam_row = class_to_row.get(1)
    if ham_row is None or spam_row is None:
        raise RuntimeError(f"Unerwartete Klassen in model.classes_: {model.classes_}")

    top_ham = np.argsort(log_probs[ham_row])[-topk:][::-1]
    top_spam = np.argsort(log_probs[spam_row])[-topk:][::-1]

    print("\nTop Wörter nach P(w|class) (typische Wörter):")
    print(f"HAM (0):  {', '.join(feature_names[i] for i in top_ham)}")
    print(f"SPAM (1): {', '.join(feature_names[i] for i in top_spam)}")


def print_top_discriminative_words(
    model: MultinomialNB,
    vectorizer: CountVectorizer,
    topk: int = 10,
) -> None:
    """
    Gibt Wörter aus, die die Klassen am stärksten unterscheiden.
    Dafür Log-Odds:
      score(word) = log P(word|spam) - log P(word|ham)
    Positive Werte -> eher SPAM, negative -> eher HAM
    """
    feature_names = vectorizer.get_feature_names_out()
    log_probs = model.feature_log_prob_

    class_to_row = {c: i for i, c in enumerate(model.classes_)}
    ham_row = class_to_row[0]
    spam_row = class_to_row[1]

    log_odds = log_probs[spam_row] - log_probs[ham_row]

    top_spam = np.argsort(log_odds)[-topk:][::-1]  # größte positive
    top_ham = np.argsort(log_odds)[:topk]          # kleinste (stark negativ)

    print("\nTop trennende Wörter (Log-Odds):")
    print(f"SPAM-lastig: {', '.join(feature_names[i] for i in top_spam)}")
    print(f"HAM-lastig:  {', '.join(feature_names[i] for i in top_ham)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="spam_ham_dataset.csv", help="Pfad zur CSV-Datei (z.B. spam.csv)")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test-Anteil (default 0.2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default 42)")
    parser.add_argument("--min_df", type=int, default=2, help="Min. Dokumenthäufigkeit für Wörter (default 2)")
    parser.add_argument("--topk", type=int, default=10, help="Anzahl Top-Wörter (default 10)")
    parser.add_argument("--alpha", type=float, default=1.0, help="Laplace-Smoothing alpha (default 1.0)")
    args = parser.parse_args()

    # Laden
    df = load_dataset(args.data)
    X_text = df["text"]
    y = df["label_num"].to_numpy()

    # Split
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    # Bag-of-Words (fit nur auf TRAIN!)
    vectorizer = CountVectorizer(
        lowercase=True,
        stop_words="english",
        min_df=args.min_df,
    )
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    print("Anzahl Dokumente:", X_train.shape[0] + X_test.shape[0])
    print("Anzahl Features (Wörter):", X_train.shape[1])
    print("Train:", X_train.shape, "Test:", X_test.shape)

    # Train Naive Bayes
    nb = MultinomialNB(alpha=args.alpha)
    nb.fit(X_train, y_train)

    # Evaluation
    y_pred = nb.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\nEvaluation:")
    print("Accuracy:", acc)
    print("Confusion Matrix:\n", cm)
    print(
        "\nClassification Report:\n",
        classification_report(y_test, y_pred, target_names=["ham", "spam"]),
    )

    # Wichtigste Begriffe
    print_top_words_by_class(nb, vectorizer, topk=args.topk)
    print_top_discriminative_words(nb, vectorizer, topk=args.topk)

    return 0


if __name__ == "__main__":
    sys.exit(main())
