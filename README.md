# Content Moderation

This is a project for CSE 3000 to explore the ethics of machine learning in content moderation. In this repository you will find scripts to process and vectorize text, and then train a multinomial Naive Bayes model in order to label the text. Our team used the dataset from Kaggle's Toxic Comment Classification Challenge.

## Set-Up

1.  Clone the repo:
    ```bash
    git clone https://github.com/WillSho1/contentModeration.git
    cd contentModeration
    ```
2.  Download the data set from Kaggle:
    [Link Toxic Comment CLassification Challenge Data](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data)
3.  From the download, move `train.csv`, `test.csv`, and `test_labels.csv` into `/data/raw`.
4.  Run the text processing script:
    ```bash
    python ./scripts/textProcessing.py
    ```
5.  Train the model, run predictions, and print accuracy:
    ```bash
    python ./scripts/main.py
    ```


