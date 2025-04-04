import pandas as pd
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy import sparse

def clean_text(text):
    # handle na
    if pd.isna(text):
        return ""
    # remove special chars
    text = re.sub(r'[^a-zA-Z\s]', '', str(text)).lower()
    return text

def load_and_clean_data(train_path, test_path, output_dir="data/clean"):
    # load raw data
    # limiting size
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # shape
    print("Train shape:", train.shape)
    print("Test shape:", test.shape)

    # clean the comments
    print("Cleaning comments...")
    train['comment_text'] = train['comment_text'].apply(clean_text)
    test['comment_text'] = test['comment_text'].apply(clean_text)

    # save cleaned data
    print("Saving cleaned data...")
    train.to_csv(f"{output_dir}/train_clean.csv", index=False)
    test.to_csv(f"{output_dir}/test_clean.csv", index=False)
    return train, test

def vectorize_data(train, test, output_dir="data/processed"):
    # ignore common terms and stopwords, lowercase, unigram/bigram, l2 normalization, sublinear tf scaling
    vectorizer = TfidfVectorizer(
        max_df=0.5,
        min_df=10,
        max_features=20000,
        stop_words='english',
        ngram_range=(1, 2),
        sublinear_tf=True,
        norm='l2'
    )

    # fit vectorizer on train and apply to both
    tfidf_train = vectorizer.fit_transform(train['comment_text'])
    tfidf_test = vectorizer.transform(test['comment_text'])
    
    # save vectorized data and vectorizer
    print("Saving vectorized data and vectorizer...")
    sparse.save_npz(f"{output_dir}/tfidf_train.npz", tfidf_train)
    sparse.save_npz(f"{output_dir}/tfidf_test.npz", tfidf_test)
    joblib.dump(vectorizer, f"{output_dir}/tfidf_vectorizer.joblib")
    
    return tfidf_train, tfidf_test, vectorizer

if __name__ == "__main__":
    train_path = "data/raw/train.csv"
    test_path = "data/raw/test.csv"
    out_clean_path = "data/clean"
    out_processed_path = "data/processed"

    # load data
    print('Loading and cleaning data...')
    train_clean, test_clean = load_and_clean_data(train_path, test_path, out_clean_path)

    # vectorize text
    print('Vectorizing text...')
    v_train, v_test, vectorizer = vectorize_data(train_clean, test_clean, out_processed_path)
    print('Done!')

    # eda
    print('Clean_train:', train_clean.shape)
    print('Clean_test:', test_clean.shape)
    
    # print information about the vectorization
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"Train shape: {v_train.shape}")
    print(f"Test shape: {v_test.shape}")
    print(f"Train sparsity: {100.0 * (1.0 - v_train.nnz / (v_train.shape[0] * v_train.shape[1]))}%")
    print(f"Test sparsity: {100.0 * (1.0 - v_test.nnz / (v_test.shape[0] * v_test.shape[1]))}%")
    print('Done!')
