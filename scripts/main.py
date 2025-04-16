import pandas as pd
import numpy as np
import scipy.sparse as sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score

def train_model_on_label(X_train, train_data, X_test, test_data, label):
    # train model on provided label
    mnb = MultinomialNB(alpha=1.0)
    print(f"Training MNB model on label: {label.upper()}...")
    mnb.fit(X_train, train_data[label])

    # predict and evaluate accuracy
    print("Predicting on train data...")
    cv_scores = cross_val_score(mnb, X_train, train_data[label], cv=5, scoring='accuracy')
    print(f"Cross-validation accuracy: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")

    print("Predicting on test data...")
    cv_scores = cross_val_score(mnb, X_test, test_data[label], cv=5, scoring='accuracy')
    print(f"Cross-validation accuracy: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")
    # Save predictions and the test labels for confusion matrix
    predictions = mnb.predict(X_test)
    true_labels = test_data[label].values

    output_dir = "data/predictions"
    np.savez_compressed(f"{output_dir}/{label}_predictions.npz",
                        y_true=true_labels, y_pred=predictions)

if __name__ == '__main__':
    # load data
    print('Loading data...')
    train_data = pd.read_csv("data/clean/train_clean.csv")
    test_data = pd.read_csv("data/clean/test_clean.csv")
    print("Loading vectorized data...")
    X_train = sparse.load_npz("data/processed/tfidf_train.npz")
    X_test = sparse.load_npz("data/processed/tfidf_test.npz")

    # compare shapes and first rows
    print(f"Train shape: {train_data.shape}")
    print(f"Test shape: {test_data.shape}")

    # iterate over labels
    labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    for label in labels:
        print(f"Label: {label}")
        # train model on each label
        train_model_on_label(X_train, train_data, X_test, test_data, label)
