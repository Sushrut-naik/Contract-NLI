# Contract NLI Project

## Overview
The project addresses the need for efficient contract review, a process which is essential to business transactions, yet time-consuming and costly. Contractual obligations and clauses often contain nuanced language, including negations and exceptions, which are challenging to interpret and automate. This project explores Document-level Natural Language Inference (NLI) to automate contract analysis, classifying hypotheses related to contract clauses as "entailed," "contradicting," or "neutral" and pinpointing supporting evidence within the contract text.

## Features
- **Baseline Models**: Implement majority vote and cosine similarity baselines for comparison.
- **Data Preparation**: Preprocess and tokenize contract documents.
- **Model Training**: Train a BERT-based NLI model with span annotations.
- **Evaluation**: Assess model performance using precision, recall, and F1 scores.

## Baseline Models 

**1. Majority Vote**
- Task: NLI only

- **Description:** The Majority Vote baseline is a simplistic approach where the model always predicts the majority class label for each hypothesis. Since this is a highly simplistic model, it assumes that the most frequent label in the training set will apply to all test cases.

**2. Doc TF-IDF + SVM (Support Vector Machine)**
- Task: NLI only

- **Description:** This baseline uses a linear SVM classifier to predict NLI labels based on document-level TF-IDF features. TF-IDF captures the importance of words in the document, while the linear SVM is a strong classifier for text-based tasks. Here, the contract document is represented as a bag of words using unigrams, and the model aims to predict the correct NLI label (entailment, contradiction, neutral).

**3. Span TF-IDF + Cosine Similarity**
- Task: Evidence identification only

- **Description:** This baseline focuses on identifying evidence spans within contracts. It computes unigram-level TF-IDF vectors for each hypothesis and compares them to each potential span in the document using cosine similarity. The span with the highest similarity score is selected as evidence for supporting or refuting the hypothesis.

**4. Span TF-IDF + SVM**
- Task: Evidence identification only

- **Description:** In this approach, a span-level linear SVM is used to classify whether a particular span in the contract is evidence supporting the hypothesis. Like the previous models, it relies on unigram bag-of-words features to represent spans and hypotheses.


## Dataset Analysis
- We have used 607 annotated contracts, the largest dataset available.
Split into : 
    - Train Set : 423 Contracts
    - Dev Set : 61 Contracts
    - Test Set : 123 Contracts

## Span NLI BERT
- The architecture includes two Multi-Layer Perceptron (MLP) classifiers:
    - **[CLS] Token Classifier:** The embedding corresponding to the [CLS] token is used for a multi-class classification task. It predicts whether the hypothesis is "entailed," "not mentioned," or "contradicted" by the document context

    - **[SPAN] Token Classifier:** The span tokens are used in a multi-label classification task through a separate MLP classifier. This classifier determines whether each span serves as evidence for any of the 17 hypotheses related to the contract. For each span, it outputs probabilities indicating its relevance to the various hypotheses.

## Execute :
```shell
For Training :
- python3 train.py

For Testing :
- python3 test.py
```

## Models

Bert-base-uncased - https://iiitaphyd-my.sharepoint.com/:u:/g/personal/ronak_patel_students_iiit_ac_in/ESfSN7g0J4ZMlT5jTk1wRNQBdfopI3zY52GH442owqdpkA?e=aDabLY

Bert-large-uncased - https://iiitaphyd-my.sharepoint.com/:u:/g/personal/ronak_patel_students_iiit_ac_in/EVHrFx5pxetIrU-_YheYl0sBhzzNLK13AZsn_PRxNlMQog?e=6CzuDD