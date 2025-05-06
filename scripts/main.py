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
    # train
    mnb.fit(X_train, train_data[label])

    # cross-validation on train data
    # split train into chunks and train each chunk and evaluate on the rest
    # new models for each chunk - does not use model trained above
    print("Cross-validation on train data...")
    cv_scores = cross_val_score(mnb, X_train, train_data[label], cv=5, scoring='f1')
    print(f"Train CV Accuracy: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")

    # actual model used here
    # predictions on test data
    print("Predicting on test data...")
    #predictions = mnb.predict(X_test)
    probs = mnb.predict_proba(X_test)[:, 1]
    predictions = (probs >= 0.3).astype(int)

    # get true values
    y_true = test_data[label].values


    # evaluation
    acc = accuracy_score(test_data[label], predictions)
    prec = precision_score(test_data[label], predictions, zero_division=0)
    rec = recall_score(test_data[label], predictions, zero_division=0)
    f1 = f1_score(test_data[label], predictions, zero_division=0)
    print(f"\n📊 Evaluation Metrics for '{label}':")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    # print number of false positives and false negatives and true positives and true negatives
    fp = np.sum((predictions == 1) & (y_true == 0))
    fn = np.sum((predictions == 0) & (y_true == 1))
    tp = np.sum((predictions == 1) & (y_true == 1))
    tn = np.sum((predictions == 0) & (y_true == 0))
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives: {tp}")
    print(f"True Negatives: {tn}\n")
    
    # save the data
    np.savez(f"data/predictions/{label}_predictions.npz", 
         y_true=y_true, 
         y_pred=predictions, 
         y_probs=probs)
    
    # return performance metrics
    performance_metrics = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'false_positives': fp,
        'false_negatives': fn,
        'true_positives': tp,
        'true_negatives': tn
    }
    return performance_metrics


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
    performance_metrics = {}
    for label in labels:
        print(f"Label: {label}")

        # calculate baseline accuracy
        baseline_acc = train_data[label].value_counts(normalize=True).max()
        print(f"Baseline Accuracy: {baseline_acc:.4f}")

        # train model on each label
        performance_metrics[label] = train_model_on_label(X_train, train_data, X_test, test_data, label)
    
    # ACCURACY
    for label in labels:
        # print accurancy
        print(f"{label} Accuracy: {performance_metrics[label]['accuracy']:.4f}")
