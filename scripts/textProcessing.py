import re
import nltk
import csv
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# download nltk data if not already downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

def clean_text(text):
    # remove special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # tokenize
    words = word_tokenize(text.lower())
    # remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    # rejoin
    return ' '.join(words)

def load_data_csv(filename):
    # read data - handling np.genfromtext issue with quotes
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            data.append(row)
    return data


if __name__ == "__main__":
    # load data
    print('Loading data...')
    train = load_data_csv('data/train.csv')
    test = load_data_csv('data/test.csv')

    # process comments
    print('Cleaning comments...')
    for row in train:
        row[1] = clean_text(row[1])
    for row in test:
        row[1] = clean_text(row[1])

    # save the cleaned data
    print('Saving cleaned data...')
    np.savetxt('data/clean/train_clean.csv', train, delimiter=',', fmt='%s')
    np.savetxt('data/clean/test_clean.csv', test, delimiter=',', fmt='%s')
