from matplotlib import pyplot as plt
import pandas as pd
import json
from utils import load_data, get_labels, cfg
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns

def get_hypothesis(data: dict) -> list:
    hypothesis = {}
    majority_ct = {}
    for key, value in data['labels'].items():
        hypothesis[key] = value['hypothesis']
        majority_ct[key] = {'NotMentioned': 0, 'Entailment': 0, 'Contradiction': 0}
    return hypothesis, majority_ct

def run_inference(test_data: dict, majority_ct: dict) -> dict:
    predictions = []
    true_labels = []
    
    for doc in test_data['documents']:
        for key, value in doc['annotation_sets'][0]['annotations'].items():
            # Get the true label for the hypothesis
            true_labels.append(value['choice'])
            predictions.append(majority_ct[key])
    
    return predictions, true_labels

def evaluate(y_pred: dict, y_true: dict):
    
    print(classification_report(y_true, y_pred))
    
    cm = confusion_matrix(y_true, y_pred, labels=['NotMentioned', 'Entailment', 'Contradiction'])
    
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['NotMentioned', 'Entailment', 'Contradiction'], yticklabels=['NotMentioned', 'Entailment', 'Contradiction'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.show()

if __name__ == "__main__":
    # Load training data and get majority vote
    train = load_data(cfg['train_path'])
    hypothesis, majority_ct = get_hypothesis(train)
    labels = get_labels()
    
    # Count majority votes from the training set
    for doc in train['documents']:
        for key, value in doc['annotation_sets'][0]['annotations'].items():
            majority_ct[key][value['choice']] += 1

    # Find the majority vote for each hypothesis
    for key, value in majority_ct.items():
        majority_ct[key] = max(value, key=value.get)
    
    # Load test data
    test = load_data(cfg['test_path'])
    
    # Run inference on test data using the majority vote
    predictions, true_labels = run_inference(test, majority_ct)
    # Evaluate predictions
    evaluate(predictions, true_labels)