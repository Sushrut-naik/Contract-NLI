import torch
from torch.utils.data import Dataset
from icecream import ic
from utils import get_labels, get_hypothesis_idx

# Configuration dictionary
cfg = {
    'raw_data_dir' : './data',
    'train_path' : 'train.json',
    'test_path' : 'test.json',
    'dev_path' : 'dev.json',
    'model_name': 'bert-base-uncased',
    'models_save_dir': '../models/',       # path for model save directory
    'dataset_dir': '../dataset/',          # path for dataset directory
    'trained_model_dir': '/19/trained_model/',
    'batch_size': 32,
    'max_length': 512,
    "results_dir": "./results",
}


# class NLIDataset(Dataset):
#     def __init__(self, documents, tokenizer, hypothesis, context_sizes, surround_character_size):
#         """
#         Dataset class for NLI task with span annotation.
        
#         Args:
#             documents (list): List of document dictionaries with annotations.
#             tokenizer (AutoTokenizer): Tokenizer for encoding text.
#             hypothesis (dict): Dictionary mapping hypothesis names to text.
#             context_sizes (list): List of context sizes for splitting documents.
#             surround_character_size (int): Overlap size between contexts.
#         """

#         #     label_dict {
#         #     'NotMentioned': 0,
#         #     'Entailment': 1,
#         #     'Contradiction': 2,
#         # }
#         label_dict = get_labels()
#         self.tokenizer = tokenizer

#         ''' List of dictionaries of all data points 
#         One data point will contain : {
#             hypothesis
#             marked_beg
#             marked_end
#             premise -> [span] text [span] text .....
#             span_ids -> []
#             span_labels -> []
#             nli_label -> for cls token
#             hypothesis_id
#             doc_id
#         }
#         {'hypotheis': "receiving party shall not reverse engineer any objects which embody disclosing party's confidential information.", 
#         'marked_beg': False, 
#         'marked_end': False, 
#         'premise': ' [SPAN] rmation with the Bidder participating in the RFP; [SPAN] WHEREAS UNHCR agrees to provide such data and information to the Bidder for the sole purpose of preparing its Proposal under said RFP; [SPAN] WHEREAS  [SPAN] the Bidder is willing to ensure that UNHCR’s data and information will be held in strict confidence and only used for the permitted purpose; [SPAN] NOW, THEREFORE, the Parties agree as follows: [SPAN] 1. “Confidential Information”, whenever used in this Agreement, shall mean any data, document, specification and other information or material, that is delivered or disclosed by UNHCR to the Recipient in any form whatsoever, whether orally, visually in writing or otherwise (including computerized form), and that, at the time of disclosure to the Recipient, is designated as confidential. [SPAN] 2. The Confidential Information that is delivered or otherwise disclosed by the Discloser to the Recipient shall be held in trust and confidence by the Recipient and shall be handled as follows: [SPAN] 2.1  [SPAN] The Recipient shall use the sam', 
#         'nli_label': tensor(0), 
#         'span_labels': tensor([0, 0, 0, 0, 0, 0, 0]), 
#         'doc_id': tensor(0), 
#         'hypothesis_id': tensor(11), 
#         'span_ids': tensor([10, 11, 12, 13, 14, 15, 16])}
#         '''

#         data_points = []
#         contexts = [{}]  # Stores context details

#         # First we are spliting document in contexts then generatig data points
#         # [1100]
#         for context_size in context_sizes:
#             # For all documents 
#             # [{doc1},{doc2} ....]
#             for i, doc in enumerate(documents):
#                 # For each doc
#                 char_idx = 0
#                 # till the end of text of one doc
#                 while char_idx < len(doc['text']):
#                     # ic(char_idx)
#                     # All annotated spans present in one doc
#                     # [[start,end],[start,end],[start,end].......] - continuous spans
#                     document_spans = doc['spans']
#                     cur_context = {
#                         'doc_id': i,
#                         # context's start
#                         'start_char_idx': char_idx,
#                         # context's end = context's start + contetSize
#                         'end_char_idx': char_idx + context_size,
#                         # All spans in this context
#                         'spans': []
#                     }
                    
#                     # For all spans of this DOCUMENT
#                     for j, (start, end) in enumerate(document_spans):
#                         # Ignoring all previous spans before current context
#                         if end <= char_idx:
#                             continue
#                         cur_context['spans'].append({
#                             # Start of span
#                             'start_char_idx': max(start, char_idx),
#                             # End of span
#                             'end_char_idx': min(end, char_idx + context_size),
#                             # If span lies fully inside current context then marked
#                             'marked': start >= char_idx and end <= char_idx + context_size,
#                             'span_id': j
#                         })
#                         # Ignoring all spans after current context
#                         if end > char_idx + context_size:
#                             break

#                     # If current context is similar to last added context then start the process from endindex of current context - surrounded size
#                     if cur_context == contexts[-1]:
#                         char_idx = cur_context['end_char_idx'] - surround_character_size
#                         continue

#                     # Adding current context in contexts
#                     contexts.append(cur_context)
#                     if len(cur_context['spans']) == 1 and not cur_context['spans'][0]['marked']:
#                         char_idx = cur_context['end_char_idx'] - surround_character_size
#                     else:
#                         # Setting new start of context as current context's last span's start index - surrounded size
#                         char_idx = cur_context['spans'][-1]['start_char_idx'] - surround_character_size

#         contexts.pop(0)

#         # hypothesis = { key : hypothesis name , val : hypothesis text}
#         # (e.g : {nda11 : text, nda10 : text, nda1 : text ....}) total 17 values
#         for nda_name, nda_desc in hypothesis.items():
#             # Contexts contains all context of all document
#             # For all contexts
#             # We can get to know doc id present inside context
#             for context in contexts:
#                 doc_id = context['doc_id']
#                 # nli_label -> ground truth -> actual label present inside dataset for current hypothesis -> Entailment/Contradiction/NotMentioned
#                 # For cls token
#                 nli_label = label_dict[documents[doc_id]['annotation_sets'][0]['annotations'][nda_name]['choice']]
#                 # Creating data points and adding all datapoints inside datapoints list
#                 # data_point = {
#                 #     'hypotheis': nda_desc,
#                 #     # First span's marked value of current context
#                 #     'marked_beg': context['spans'][0]['marked'],
#                 #     # Last span's marked value of current context
#                 #     'marked_end': context['spans'][-1]['marked'],
#                 #     'premise': '',
#                 #     'doc_id': torch.tensor(doc_id, dtype=torch.long),
#                 #     # nda11 -> then 11 is id
#                 #     'hypothesis_id': torch.tensor(get_hypothesis_idx(nda_name), dtype=torch.long),
#                 #     # nli label -> ground truth for cls token ->entailment,contradiction,not mentioned
#                 #     'nli_label': torch.tensor(nli_label, dtype=torch.long)
#                 # }
#                 data_point = {}
#                 data_point['hypotheis'] = nda_desc
#                 cur_premise = ""
#                 data_point['marked_beg'] = context['spans'][0]['marked']
#                 data_point['marked_end'] = context['spans'][-1]['marked']
#                 doc_id = context['doc_id']
#                 hypothesis_id = get_hypothesis_idx(nda_name)
#                 span_ids = []

#                 # If only one span then marking end as true
#                 if len(context['spans']) == 1:
#                     data_point['marked_end'] = True

#                 # label map to span id -> for each span id whether it is evidence of current hypothesis or not
#                 span_labels = []
#                 span_ids = []

#                 # For all spans of current context.
#                 for span in context['spans']:

#                     # Here val is 0 or 1 according to dataset
#                     val = int(span['span_id'] in documents[doc_id]['annotation_sets'][0]['annotations'][nda_name]['spans'])
#                     # making 0 -> -1 and 1 -> 1 -> for sigmoid
#                     val = 2 * val - 1

#                     # taking only fully covered spans only so that no inconsistency happens
#                     # Suppose a span begins in one context and ends in another. If it were included in both contexts:
#                     # The label might apply inconsistently, as part of the span could be labeled as evidence while another part isn’t.
#                     if span['marked']:
#                         span_labels.append(val)
#                         span_ids.append(span['span_id'])
                    
#                     # cur_premise =>  [SPAN] text [SPAN] text [SPAN] text
#                     data_point['premise'] += ' [SPAN] ' + documents[doc_id]['text'][span['start_char_idx']:span['end_char_idx']]

#                 # Setting all spans ids corresponding span value as 0
#                 if nli_label == label_dict['NotMentioned']:
#                     span_labels = torch.zeros(len(span_labels), dtype=torch.long)

#                 # 1 or -1 for each span id
#                 # Length of span id and span label will be same
#                 data_point['span_labels'] = torch.tensor(span_labels, dtype=torch.long)
#                 data_point['span_ids'] = torch.tensor(span_ids, dtype=torch.long)

#                 # add one data point in all data points
#                 data_points.append(data_point)
#                 # ic(data_point)

#         self.data_points = data_points
#         self.span_token_id = self.tokenizer.convert_tokens_to_ids('[SPAN]')

#     def __len__(self):
#         return len(self.data_points)

#     def __getitem__(self, idx):
#         tokenized_data = self.tokenizer(
#             [self.data_points[idx]['hypotheis']],
#             [self.data_points[idx]['premise']],
#             padding='max_length',
#             truncation=True,
#             return_tensors='pt'
#         )

#         tokenized_data['input_ids'] = tokenized_data['input_ids'].squeeze()
#         tokenized_data['attention_mask'] = tokenized_data['attention_mask'].squeeze()
#         tokenized_data['token_type_ids'] = tokenized_data['token_type_ids'].squeeze()

#         span_indices = torch.where(tokenized_data['input_ids'] == self.span_token_id)[0]
#         if not self.data_points[idx]['marked_beg']:
#             span_indices = span_indices[1:]
#         if not self.data_points[idx]['marked_end'] or tokenized_data['attention_mask'][-1] == 0:
#             span_indices = span_indices[:-1]

#         span_ids = self.data_points[idx]['span_ids'][:len(span_indices)]

#         return {
#             'input_ids': tokenized_data['input_ids'],
#             'attention_mask': tokenized_data['attention_mask'],
#             'token_type_ids': tokenized_data['token_type_ids'],
#             'span_indices': span_indices,
#             'nli_label': self.data_points[idx]['nli_label'],
#             'span_labels': self.data_points[idx]['span_labels'][:len(span_indices)],
#             'data_for_metrics': {
#                 'doc_id': self.data_points[idx]['doc_id'],
#                 'hypothesis_id': self.data_points[idx]['hypothesis_id'],
#                 'span_ids': span_ids
#             }
#         }



from torch.utils.data import Dataset
import random
import torch

class NLIDataset(Dataset):
    def __init__(self, documents, tokenizer, hypothesis, context_sizes, surround_character_size):
        label_dict = get_labels()
        self.tokenizer = tokenizer

        self.tokenizer.add_special_tokens({'additional_special_tokens': ['[SPAN]']})

        data_points = []
        contexts = [{}]

        for context_size in context_sizes:
            for i, doc in enumerate(documents):
                char_idx = 0
                while char_idx < len(doc['text']):
                    # ic(char_idx)
                    document_spans = doc['spans']
                    cur_context = {
                        'doc_id': i,
                        'start_char_idx': char_idx,
                        'end_char_idx': char_idx + context_size,
                        'spans' : [],
                    }

                    for j, (start, end) in enumerate(document_spans):
                        if end <= char_idx:
                            continue

                        cur_context['spans'].append({
                            'start_char_idx': max(start, char_idx),
                            'end_char_idx': min(end, char_idx + context_size),
                            'marked': start >= char_idx and end <= char_idx + context_size,
                            'span_id': j
                        })

                        if end > char_idx + context_size:
                            break

                    if cur_context == contexts[-1]:
                        char_idx = cur_context['end_char_idx'] - surround_character_size
                        continue

                    contexts.append(cur_context)
                    if len(cur_context['spans']) == 1 and not cur_context['spans'][0]['marked']:
                        char_idx = cur_context['end_char_idx'] - surround_character_size
                    else:
                        char_idx = cur_context['spans'][-1]['start_char_idx'] - surround_character_size

        contexts.pop(0)

        for nda_name, nda_desc in hypothesis.items():
            for i, context in enumerate(contexts):

                nli_label = label_dict[documents[context['doc_id']]['annotation_sets'][0]['annotations'][nda_name]['choice']]

                data_point = {}
                data_point['hypotheis'] = nda_desc
                cur_premise = ""
                data_point['marked_beg'] = context['spans'][0]['marked']
                data_point['marked_end'] = context['spans'][-1]['marked']
                doc_id = context['doc_id']
                hypothesis_id = get_hypothesis_idx(nda_name)
                span_ids = []

                if len(context['spans']) == 1:
                    data_point['marked_end'] = True

                span_labels = []

                for span in context['spans']:
                    val = int(span['span_id'] in documents[context['doc_id']]['annotation_sets'][0]['annotations'][nda_name]['spans'])

                    val = 2 * val - 1 # making 0 -> -1 and 1 -> 1

                    if span['marked']:
                        span_labels.append(val)
                        span_ids.append(span['span_id'])

                    cur_premise += ' [SPAN] '
                    cur_premise += documents[context['doc_id']]['text'][span['start_char_idx']:span['end_char_idx']]

                data_point['premise'] = cur_premise
                
                if nli_label == get_labels()['NotMentioned']:
                    span_labels = torch.zeros(len(span_labels), dtype=torch.long)

                data_point['nli_label'] = torch.tensor(nli_label, dtype=torch.long)
                data_point['span_labels'] = torch.tensor(span_labels, dtype=torch.long)
                data_point['doc_id'] = torch.tensor(doc_id, dtype=torch.long)
                data_point['hypothesis_id'] = torch.tensor(hypothesis_id, dtype=torch.long)
                data_point['span_ids'] = torch.tensor(span_ids, dtype=torch.long)

                data_points.append(data_point)

        self.data_points = data_points
        self.span_token_id = self.tokenizer.convert_tokens_to_ids('[SPAN]')

    def __len__(self):
        return len(self.data_points)

    def __getitem__(self, idx):
        tokenized_data = self.tokenizer(
            [self.data_points[idx]['hypotheis']],
            [self.data_points[idx]['premise']],
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        tokenized_data['input_ids'] = tokenized_data['input_ids'].squeeze()
        tokenized_data['attention_mask'] = tokenized_data['attention_mask'].squeeze()
        tokenized_data['token_type_ids'] = tokenized_data['token_type_ids'].squeeze()

        span_indices = torch.where(tokenized_data['input_ids'] == self.span_token_id)[0]

        if not self.data_points[idx]['marked_beg']:
            span_indices = span_indices[1:]
        
        if not self.data_points[idx]['marked_end'] or tokenized_data['attention_mask'][-1] == 0:
            span_indices = span_indices[:-1]
        
        span_ids = self.data_points[idx]['span_ids']
        span_ids = span_ids[:len(span_indices)]

        return {
            'input_ids': tokenized_data['input_ids'],
            'attention_mask': tokenized_data['attention_mask'],
            'token_type_ids': tokenized_data['token_type_ids'],
            'span_indices': span_indices,
            'nli_label': self.data_points[idx]['nli_label'],
            'span_labels': self.data_points[idx]['span_labels'][:len(span_indices)],
            'data_for_metrics': {
                'doc_id': self.data_points[idx]['doc_id'],
                'hypothesis_id': self.data_points[idx]['hypothesis_id'],
                'span_ids': span_ids,
            }
        }