import pandas as pd
import numpy as np
import scipy.sparse as sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import cross_val_score

def train_model_on_label(X_train, train_data, X_test, test_data, label):
    mnb = MultinomialNB(alpha=1.0)
    print(f"Training MNB model on label: {label.upper()}...")
    mnb.fit(X_train, train_data[label])

    #Train cross-validation
    print("Cross-validation on train data...")
    cv_scores = cross_val_score(mnb, X_train, train_data[label], cv=5, scoring='accuracy')
    print(f"Train CV Accuracy: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")

    #Test predictions
    print("Predicting on test data...")
    predictions = mnb.predict(X_test)
    true_labels = test_data[label].values

    #Save predictions
    # output_dir = "data/predictions"
    # #os.makedirs(output_dir, exist_ok=True)
    # np.savez_compressed(f"{output_dir}/{label}_predictions.npz", y_true=true_labels, y_pred=predictions)

    # Evaluation metrics
    acc = accuracy_score(true_labels, predictions)
    prec = precision_score(true_labels, predictions, zero_division=0)
    rec = recall_score(true_labels, predictions, zero_division=0)
    f1 = f1_score(true_labels, predictions, zero_division=0)

    print(f"\n📊 Evaluation Metrics for '{label}':")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}\n")

    #Optional: Full classification report
    #print("Classification Report:")
    #print(classification_report(true_labels, predictions))

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
