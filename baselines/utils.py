import os
import json
from nltk import word_tokenize 
import re

import os

# Ensure correct base directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


cfg = {
    'raw_data_dir': os.path.join(BASE_DIR, 'data/'),
    'train_path': os.path.join(BASE_DIR, 'data/train.json'),
    'test_path': os.path.join(BASE_DIR, 'data/test.json'),
    'dev_path': os.path.join(BASE_DIR, 'data/dev.json'),
    'a': "a"
}

def load_data(path: str) -> json:
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def get_labels() -> dict:
    return {
        'NotMentioned': 0,
        'Entailment': 1,
        'Contradiction': 2,
    }

def get_hypothesis(data: dict) -> list:
    hypothesis = {}
    for key, value in data['labels'].items():
        hypothesis[key] = clean_str(value['hypothesis'])
    return hypothesis

def tokenize(str: str) -> str:
    return ' '.join(word_tokenize(str))

def clean_str(str: str) -> str:
    # remove '\n' character
    str = str.replace('\n', ' ')
    # remove '\t' character
    str = re.sub(r'\\t', ' ', str)
    # remove '\r' character
    str = re.sub(r'\\r', ' ', str)
    # remove more than 2 consecutive occcurance of a character
    str = re.sub(r'(.)\1{2,}', r'\1', str)
    return str.strip().lower()