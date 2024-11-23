import os
import json
import re
import logging as log
from nltk import word_tokenize
from sklearn.metrics import f1_score
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_curve
from icecream import ic

# Configure logging for debugging
log.basicConfig(level=log.DEBUG)

def get_file_path(self, file_key: str) -> str:
    """Returns full path for a given file key."""
    paths = {
        'train': self.train_path,
        'test': self.test_path,
        'dev': self.dev_path
    }
    return os.path.join(self.raw_data_dir, paths.get(file_key, ''))

def load_data(file_path: str) -> dict:
    """Loads JSON data from a specified file path."""
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        log.debug(f"Data loaded successfully from {file_path}")
        return data
    except Exception as e:
        log.error(f"Error loading JSON data: {e}")
        return {}
    

def get_labels() -> dict:
    """Returns a dictionary of label mappings."""
    return {
        'NotMentioned': 0,
        'Entailment': 1,
        'Contradiction': 2,
    }

def clean_text(text: str) -> str:
    """Cleans a given text string by removing unwanted characters and formatting."""
    text = text.replace('\n', ' ')
    text = re.sub(r'\\t', ' ', text)  # Remove tab characters
    text = re.sub(r'\\r', ' ', text)  # Remove carriage return characters
    text = re.sub(r'(.)\1{2,}', r'\1', text)  # Replace 3+ repeated characters
    return text.strip().lower()


def tokenize_text(text: str) -> str:
    """Tokenizes a given text using NLTK word tokenization and joins tokens with spaces."""
    tokens = word_tokenize(text)
    return ' '.join(tokens)

def get_hypotheses(data: dict) -> dict:
    """Returns a dictionary of cleaned hypotheses from given labeled data."""
    hypotheses = {}
    labels = data.get('labels', {})
    for key, value in labels.items():
        hypotheses[key] = clean_text(value.get('hypothesis', ''))
    log.debug(f"Hypotheses extracted: {list(hypotheses.items())[:5]}")
    return hypotheses

def get_hypothesis_idx(hypothesis_name: str) -> int:
    """Extracts numerical index from a hypothesis name string."""
    try:
        return int(hypothesis_name.split('-')[-1])
    except ValueError:
        log.error(f"Invalid hypothesis name format: {hypothesis_name}")
        return -1
    

def get_micro_average_precision_at_recall(y_true, y_pred, recall_level):
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    return np.interp(recall_level, recall[::-1], precision[::-1])


def calculate_micro_average_precision(y_true, y_pred):
    """Calculate the micro average precision score.

    Args:
        y_true (np.array): True labels.
        y_pred (np.array): Predicted labels.

    Returns:
        float: Micro average precision score.
    """
    # Get the number of classes
    num_classes = len(np.unique(y_true))
    
    if num_classes == 0:
        return 0.0

    # initialize the average precision score
    average_precision = 0.0

    # loop over all classes
    for class_idx in range(num_classes):
        # get the indices for this class
        y_true_indices = np.where(y_true == class_idx)
        # calculate the average precision score for this class
        average_precision += ic(precision_score(
            y_true[y_true_indices], y_pred[y_true_indices], average="micro"
        ))

    # return the average over all classes
    return average_precision / num_classes

def calculate_f1_score_for_class(y_true, y_pred, class_idx):
    """Calculate the F1 score for a given class.

    Args:
        y_true (np.array): True labels.
        y_pred (np.array): Predicted labels.
        class_idx (int): Index of the class.

    Returns:
        float: F1 score for the given class.
    """
    # get the indices for the given class
    y_true_indices = np.where(y_true == class_idx)
    # calculate the F1 score for the given class
    return f1_score(
        y_true[y_true_indices], y_pred[y_true_indices], average="macro"
    )


def precision_at_recall(y_true, y_scores, recall_threshold):
    precision, recall, threshold = precision_recall_curve(y_true, y_scores)
    idx = (np.abs(recall - recall_threshold)).argmin()  # Find nearest recall value to threshold
    ic(threshold[idx])
    return precision[idx]
