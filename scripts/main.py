import pandas as pd
import scipy.sparse as sparse
from sklearn.feature_extraction.text import TfidfVectorizer

if __name__ == '__main__':
    # load data
    print('Loading data...')
    train_data = pd.read_csv("data/clean/train_clean.csv")
    test_data = pd.read_csv("data/clean/test_clean.csv")
    X_train = sparse.load_npz("data/processed/tfidf_train.npz")
    X_test = sparse.load_npz("data/processed/tfidf_test.npz")

    train_meta = train_data.drop(columns=['comment_text'])
    test_meta = test_data.drop(columns=['comment_text'])

    # need to combine test_labels with test_data