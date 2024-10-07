# <a name="_w7wxzktdq5sr"></a>**Baseline Models**
### <a name="_qavhhky2tfvh"></a>**1. Majority Vote**
**Task**: NLI only

**Description**: The Majority Vote baseline is a simplistic approach where the model always predicts the majority class label for each hypothesis. Since this is a highly simplistic model, it assumes that the most frequent label in the training set will apply to all test cases.

**Analysis**: While this method provides a useful reference point for model performance, it tends to produce biased results in datasets where label distribution is unbalanced. This is especially evident in NLI tasks, where certain labels (like neutral or entailment) may dominate. Hence, Majority Vote is expected to perform poorly compared to other models due to its lack of sophistication.



**Performance Metrics:**

|**Label**|**Precision**|**Recall**|**F1-Score**|**Support**|
| - | - | - | - | - |
|Contradiction|0\.48|0\.54|0\.51|220|
|Entailment|0\.68|0\.78|0\.73|968|
|NotMentioned|0\.68|0\.56|0\.61|903|
|Accuracy|||0\.66|2091|
|Macro Avg|0\.62|0\.63|0\.62|2091|
|Weighted Avg|0\.66|0\.66|0\.66|2091|

**Performance Analysis**:

- The model performs okay in terms of both precision and recall, especially for the "Entailment" and "NotMentioned" labels, as shown by the high F1-scores.
- However, for the "Contradiction" label, there is some class imbalance, with only one sample leading to a lower precision but perfect recall. This indicates that the majority vote method could be overfitting to prevalent labels due to the small dataset size.


### <a name="_gbr48d4jivpu"></a>**2. Doc TF-IDF + SVM (Support Vector Machine)**
**Task**: NLI only

**Description**: This baseline uses a linear SVM classifier to predict NLI labels based on document-level TF-IDF features. TF-IDF captures the importance of words in the document, while the linear SVM is a strong classifier for text-based tasks. Here, the contract document is represented as a bag of words using unigrams, and the model aims to predict the correct NLI label (entailment, contradiction, neutral).

**Analysis**: The Doc TF-IDF + SVM baseline is a classic choice for NLI tasks, and it has shown reasonable success in various domains. However, it does not capture more sophisticated semantic relationships beyond surface-level word frequencies. For contract language, which often includes complex clauses and legal jargon, this method might miss deeper contextual meanings, leading to moderate performance on NLI tasks. Compared to Majority Vote, this model is more robust but still limited by its reliance on simple bag-of-words features.



**Performance Metrics:**

|**Label**|**Precision**|**Recall**|**F1-Score**|**Support**|
| - | - | - | - | - |
|Contradiction|0\.70|0\.62|0\.66|903|
|Entailment|0\.72|0\.77|0\.74|968|
|NotMentioned|0\.48|0\.54|0\.51|220|
|Accuracy|||0\.68|2091|
|Macro Avg|0\.63|0\.64|0\.64|2091|
|Weighted Avg|0\.68|0\.68|0\.68|2091|

**Performance Analysis:**

- The performance on a larger dataset drops, with a decrease in both accuracy (0.68) and F1-score (macro avg 0.64). This could be attributed to the complexity of the dataset, where the model struggles with "Contradiction" as seen by the low F1-score (0.51).
- The precision and recall for "Contradiction" are notably lower, showing difficulty in distinguishing contradictions from the other two classes. On the other hand, the "Entailment" class performs comparatively better with higher precision (0.72) and recall (0.77), showing the model's strength in predicting entailments.
- The weighted average F1-score (0.68) reflects the imbalanced nature of the dataset and better performance on the more frequent classes like "Entailment."

### <a name="_c2j2tgx6d0pg"></a>**3. Span TF-IDF + Cosine Similarity**
**Task**: Evidence identification only

**Description**: This baseline focuses on identifying evidence spans within contracts. It computes unigram-level TF-IDF vectors for each hypothesis and compares them to each potential span in the document using cosine similarity. The span with the highest similarity score is selected as evidence for supporting or refuting the hypothesis.

**Analysis**: This model is purely similarity-based and does not leverage any learned classification, relying entirely on TF-IDF for identifying spans. Cosine similarity can highlight the most lexically similar spans, but this method might fail in cases where the most relevant evidence involves paraphrasing or indirect language. While it is lightweight and easy to implement, the performance might be lower due to its lack of deep understanding of contract-specific nuances.



**Performance Metrics:**

Precision @ 80% recall:  0.030058717670690627

Mean Average Precision:  0.0461261085908014
### <a name="_sy8fxbh2hucl"></a>**4. Span TF-IDF + SVM**
**Task**: Evidence identification only

**Description**: In this approach, a span-level linear SVM is used to classify whether a particular span in the contract is evidence supporting the hypothesis. Like the previous models, it relies on unigram bag-of-words features to represent spans and hypotheses.

**Analysis**: Span TF-IDF + SVM is an improvement over Cosine Similarity as it introduces a more discriminative classifier (SVM) to determine which spans are relevant evidence. The span-level SVM can learn patterns in the dataset that go beyond surface similarity, improving the accuracy of evidence identification. However, similar to other TF-IDF-based models, it still struggles to capture deeper semantic meanings. This method may provide better results than Span TF-IDF + Cosine, especially in distinguishing subtle differences between evidence and non-evidence spans. The performance below is lower than cosine similarity as the model was trained on only 100 samples as opposed to the entire training set in cosine similarity. This was done due to very high training times for the svm approach.

**Performance Metrics:**

Precision @ 80% recall:  0.02521623982193672

Mean Average Precision:  0.02521623982193672

### Asumptions 
- Due to compute issues in span_tf_idf_svm baseline we have trained 100 contracts and done inference of 100 contracts which took around 4 hours.