import scripts.textProcessing as tp
import numpy as np



if __name__ == '__main__':
    # load data
    train = np.genfromtxt('data/train.csv', delimiter=',', dtype=None, encoding='utf-8')[1:]
    test = np.genfromtxt('data/test.csv', delimiter=',', dtype=None, encoding='utf-8')[1:]
    test_labels = np.genfromtxt('data/test_labels.csv', delimiter=',', dtype=None, encoding='utf-8')[1:]

    # process the comments
    # call tp.clean_text on each comment in train and test
    # store results in train_clean and test_clean.csv