from transformers import AutoTokenizer, AutoModelForMaskedLM, TrainingArguments, Trainer
import logging as log
from icecream import ic
import sys
import os
from pathlib import Path
import torch
from utils import *
from prepare_dataset import cfg,NLIDataset
from train import ContractNLITrainer
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from tqdm import tqdm
from contract_nli_model import ContractNLI

log.basicConfig(level=log.DEBUG)

sys.path.append('../')

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

os.environ['WANDB_ENTITY'] = 'contract-nli-db'
os.environ['WANDB_PROJECT'] = 'contract-nli-metric'
os.environ['WANDB_LOG_MODEL'] = 'end'

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
DEVICE

# cfg['model_name'] = 'nlpaueb/legal-bert-base-uncased'
# cfg['model_name'] = 'bert-base-uncased'
cfg['model_name'] = 'distilbert/distilbert-base-uncased'
cfg['batch_size'] = 32


# create dir if not exists
Path(cfg["models_save_dir"]).mkdir(parents=True, exist_ok=True)
Path(cfg["dataset_dir"]).mkdir(parents=True, exist_ok=True)


tokenizer = AutoTokenizer.from_pretrained(cfg['model_name'])

dev_data = load_data(os.path.join(cfg['raw_data_dir'], cfg['dev_path']))
test_data = load_data(os.path.join(cfg['raw_data_dir'], cfg['test_path']))

hypothesis = get_hypotheses(dev_data)

dev_data = dev_data['documents']
test_data = test_data['documents']

dev_data = dev_data[:50]
test_data = test_data[:50]

ic.disable()

ic(len(dev_data), len(test_data))
dev_dataset = NLIDataset(dev_data, tokenizer, hypothesis, [1100], 50)
test_dataset = NLIDataset(test_data, tokenizer, hypothesis, [1100], 50)

ic.enable()

del dev_data
del test_data
del hypothesis


print(len(dev_dataset))
print(len(test_dataset))

training_args = TrainingArguments(
    auto_find_batch_size=True,
    output_dir=cfg['results_dir'],   # output directory
    num_train_epochs=10,            # total number of training epochs
    gradient_accumulation_steps=4,   # number of updates steps to accumulate before performing a backward/update pass
    logging_strategy='epoch',
    # eval_steps=0.25,
    # save_steps=0.25,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    save_total_limit=2,
    load_best_model_at_end=True,
    # fp16=True,
    label_names=['nli_label', 'span_labels', 'data_for_metrics'],
    report_to='none',
)


cfg['trained_model_dir']


# api = wandb.Api()
# artifact = api.artifact('contract-nli-db/contract-nli/model-r0p4yqnz:v0', type='model')
# artifact_dir = artifact.download(cfg['trained_model_dir'])

# Define the directory where your model is stored locally.
artifact_dir = '/home/viraj/IIITH/sem3/ANLP/Project/Final/19/checkpoint-21339.zip'  # Replace this with the actual path

# Load the model directly from the local directory
model = ContractNLI.from_pretrained(artifact_dir).to(DEVICE)


# artifact_dir


class ContractNLIMetricTrainer(ContractNLITrainer):
    def __init__(self, *args, data_collator=None, **kwargs):
        super().__init__(*args, data_collator=data_collator, **kwargs)

    def evaluate(self, eval_dataset=None, ignore_keys=None):
        self.model.eval()
        self.dataloader = ic(self.get_eval_dataloader(eval_dataset))

        eval_nli_labels = []
        eval_nli_preds = []
        true_labels_per_span = {}
        probs_per_span = {}

        nli_metrics = {}

        for inputs in tqdm(self.dataloader):
            inputs = self._prepare_inputs(inputs)
            span_labels = inputs.pop('span_labels')
            nli_labels = inputs.pop('nli_label')
            data_for_metrics = inputs.pop('data_for_metrics')

            span_indices_to_consider = torch.where(span_labels != -1)[0]

            with torch.no_grad():
                outputs = self.model(**inputs)
                span_logits, nli_logits = outputs[0], outputs[1]

                span_labels = span_labels.float()
                span_logits = span_logits.float()
                
                span_labels = span_labels.view(-1)
                span_logits = span_logits.view(-1)

                # start_index = 0
                
                indices_considered = 0 # total number of span indices considered

                # find the corresponding span index in data_for_metrics['span_ids'] considering -1 to be padding index
                # ic(span_index)
                for i, span_index_row in enumerate(data_for_metrics['span_ids']):
                    current_index = 0 # current row's first -1 index
                    ic(span_index_row)
                    first_minus_one_index = torch.where(span_index_row == -1)[0]
                    ic(first_minus_one_index)
                    if len(first_minus_one_index) == 0:
                        first_minus_one_index = len(span_index_row)
                    else:
                        first_minus_one_index = first_minus_one_index[0].item()

                    key = str(data_for_metrics['doc_id'][i].item())+ '-' + str(data_for_metrics['hypothesis_id'][i].item())

                    # mask span_labels and span_logits for the current row
                    mask = span_labels[indices_considered:indices_considered+first_minus_one_index] != -1
                    span_logits_masked = span_logits[indices_considered:indices_considered+first_minus_one_index][mask]

                    spans_contribution = torch.sum(torch.sigmoid(span_logits_masked)) / (len(span_logits_masked)) 

                    if key in nli_metrics:
                        nli_metrics[key]['spans_contribution'].append(spans_contribution)
                        nli_metrics[key]['nli_logits'].append(nli_logits[i])
                    else:
                        nli_metrics[key] = {}
                        nli_metrics[key]['true_nli_labels'] = nli_labels[i]
                        nli_metrics[key]['spans_contribution'] = [spans_contribution]
                        nli_metrics[key]['nli_logits'] = [nli_logits[i]]
                    
                    current_index = first_minus_one_index
                    indices_considered += current_index
                    
                    ic(indices_considered)
                    ic(current_index)
                    cnt = 0 # count to keep track of the number of span indices added in dictionary
                    
                    for span_index in span_indices_to_consider:

                        if span_index < indices_considered:
                            cnt += 1
                            value_index = span_index - (indices_considered - current_index)
                            doc_id = data_for_metrics['doc_id'][i]
                            hypothesis_id = data_for_metrics['hypothesis_id'][i]
                            span_id = data_for_metrics['span_ids'][i][value_index]
                            key = str(doc_id)+ '-' + str(hypothesis_id)+ '-' + str(span_id)
                            true_labels_per_span[key] = span_labels[span_index]
                            if key in probs_per_span:
                                probs_per_span[key].append(torch.sigmoid(span_logits[span_index]))
                                # probs_per_span[key].append(span_logits[value_index])
                            else:
                                probs_per_span[key] = [torch.sigmoid(span_logits[span_index])]
                                # probs_per_span[key] = [span_logits[value_index]]
                        else: 
                            break 
                    
                    span_indices_to_consider = span_indices_to_consider[cnt:]

                # eval_span_preds = torch.tensor(eval_span_preds.squeeze(1), dtype=torch.long)

                nli_preds = torch.argmax(torch.softmax(nli_logits, dim=1), dim=1)
                eval_nli_labels.extend(nli_labels.cpu().numpy())
                eval_nli_preds.extend(nli_preds.cpu().numpy())

        eval_span_labels = []
        eval_span_preds = []

        for key in true_labels_per_span:
            eval_span_labels.append(true_labels_per_span[key].item())
            eval_span_preds.append(torch.mean(torch.stack(probs_per_span[key])).item())

        ##### For NLI probablities #####

        # for key in nli_metrics:
        #     nli_metrics[key]['nli_logits'] = torch.stack(nli_metrics[key]['nli_logits'])
        #     nli_metrics[key]['spans_contribution'] = torch.stack(nli_metrics[key]['spans_contribution'])

        #     span_sum = torch.sum(nli_metrics[key]['spans_contribution'])
        #     spans_contribution = nli_metrics[key]['spans_contribution'].transpose(0, -1) @ nli_metrics[key]['nli_logits']

        #     eval_nli_preds.append(torch.argmax(torch.softmax(spans_contribution/span_sum, dim=0)).item())
        #     eval_nli_labels.append(nli_metrics[key]['true_nli_labels'].item())

        ##### END #####

        eval_nli_acc = accuracy_score(eval_nli_labels, eval_nli_preds)

        ic.enable()
        ic(list(zip(eval_span_labels, eval_span_preds)))
        ic(len(eval_span_labels), len(eval_span_preds))
        ic(sum(eval_span_labels), sum(eval_span_preds))

        # find threshold for 80% recall
        # precision, recall, thresholds = precision_recall_curve(eval_span_labels, eval_span_preds)


        mAP = (average_precision_score(eval_span_labels, eval_span_preds, pos_label=0) + average_precision_score(eval_span_labels, eval_span_preds, pos_label=1))/2

        # mAP = average_precision_score(torch.tensor(true_span_labels), torch.tensor(pred_span_labels))
        precision_at_80_recall = precision_at_recall(torch.tensor(eval_span_labels), torch.tensor(eval_span_preds), 0.8)
        f1_score_for_entailment = calculate_f1_score_for_class(torch.tensor(eval_nli_labels), torch.tensor(eval_nli_preds), get_labels()['Entailment'])
        f1_score_for_contradiction = calculate_f1_score_for_class(torch.tensor(eval_nli_labels), torch.tensor(eval_nli_preds), get_labels()['Contradiction'])
        
        return {
            'mAP' : mAP,
            'precision_at_80_recall' : precision_at_80_recall,
            'nli_acc': eval_nli_acc,
            'f1_score_for_entailment': f1_score_for_entailment,
            'f1_score_for_contradiction': f1_score_for_contradiction
        }


trainer = ContractNLIMetricTrainer(
    model=model,                          
    args=training_args,                  
    eval_dataset=dev_dataset,            
    data_collator=ContractNLIMetricTrainer.collate_fn,
)


ic.disable()

results = trainer.evaluate()
results

