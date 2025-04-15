import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_confusion(label):
    # Load predictions
    data = np.load(f"data/predictions/{label}_predictions.npz")
    y_true = data['y_true']
    y_pred = data['y_pred']

    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not " + label, label])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - {label}")
    plt.savefig(f"data/predictions/{label}_confusion_matrix.png")
    plt.show()

if __name__ == '__main__':
    labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    for label in labels:
        print(f"Plotting confusion matrix for: {label}")
        plot_confusion(label)
