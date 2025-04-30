import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (precision_recall_curve, accuracy_score, 
                           precision_score, recall_score, f1_score)

def calculate_metrics_for_all_labels(labels):
    # store metrics for all labels
    metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }
    
    # calculate metrics for each label
    for label in labels:
        data = np.load(f"data/predictions/{label}_predictions.npz")
        y_true = data['y_true']
        y_pred = data['y_pred']
        
        metrics['accuracy'].append(accuracy_score(y_true, y_pred))
        metrics['precision'].append(precision_score(y_true, y_pred))
        metrics['recall'].append(recall_score(y_true, y_pred))
        metrics['f1'].append(f1_score(y_true, y_pred))
    
    return metrics

def plot_metrics_bar_charts(labels, metrics):
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    metric_keys = ['accuracy', 'precision', 'recall', 'f1']
    
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs = axs.flatten()
    
    for i, (name, key) in enumerate(zip(metric_names, metric_keys)):
        # create bar chart
        bars = axs[i].bar(labels, metrics[key], color='#ADD8E6')
        axs[i].set_title(f'{name} by Category')
        axs[i].set_ylim(0, 1.0)
        axs[i].set_ylabel(name)
        axs[i].set_xlabel('Category')
        
        # rotate x-axis labels for better readability
        plt.setp(axs[i].get_xticklabels(), rotation=45, ha='right')
        
        # add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            axs[i].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig("data/predictions/all_metrics_comparison.png")
    plt.show()

def plot_individual_pr_curves(labels):
    # create a single figure with multiple PR curves
    plt.figure(figsize=(10, 8))
    
    for label in labels:
        data = np.load(f"data/predictions/{label}_predictions.npz")
        if 'y_probs' in data:
            y_true = data['y_true']
            probs = data['y_probs']
            precision, recall, _ = precision_recall_curve(y_true, probs)
            plt.plot(recall, precision, label=label)
    
    plt.title('Precision-Recall Curves')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('data/predictions/all_pr_curves.png')
    plt.show()

if __name__ == '__main__':
    labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    # calculate metrics for all labels
    print("Calculating metrics for all labels...")
    metrics = calculate_metrics_for_all_labels(labels)
    
    # plot bar charts for all metrics
    print("Creating bar charts comparing all labels...")
    plot_metrics_bar_charts(labels, metrics)
    
    # optionally plot PR curves if available
    print("Creating combined PR curves...")
    plot_individual_pr_curves(labels)
