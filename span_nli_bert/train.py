import torch
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback, AutoTokenizer
from sklearn.utils.class_weight import compute_class_weight
from contract_nli_model import ContractNLI, ContractNLIConfig
from prepare_dataset import cfg
from prepare_dataset import NLIDataset
import os
from utils import load_data, get_hypotheses
import numpy as np
from pathlib import Path
import sys

# Add the project root to sys.path
sys.path.append('../')


# Set environment variables for W&B
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['WANDB_ENTITY'] = 'contract-nli-db'
os.environ['WANDB_PROJECT'] = 'contract-nli'
os.environ['WANDB_LOG_MODEL'] = 'end'

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("=================", DEVICE)
from pathlib import Path
Path(cfg["models_save_dir"]).mkdir(parents=True, exist_ok=True)
Path(cfg["dataset_dir"]).mkdir(parents=True, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(cfg['model_name'])

tokenizer.save_pretrained(cfg['models_save_dir'])

tokenizer = AutoTokenizer.from_pretrained(cfg['models_save_dir'])
tokenizer.add_special_tokens({'additional_special_tokens': ['[SPAN]']})


train_data = load_data(os.path.join(cfg['raw_data_dir'], cfg['train_path']))
dev_data = load_data(os.path.join(cfg['raw_data_dir'], cfg['dev_path']))
test_data = load_data(os.path.join(cfg['raw_data_dir'], cfg['test_path']))

hypothesis = get_hypotheses(train_data)

train_data = train_data['documents']
dev_data = dev_data['documents']
test_data = test_data['documents']

train_data = train_data[:2]
dev_data = dev_data[:2]
test_data = test_data[:2]

# ic.disable()

print(len(train_data), len(dev_data), len(test_data))

train_dataset = NLIDataset(train_data, tokenizer, hypothesis, [1100], 50)
dev_dataset = NLIDataset(dev_data, tokenizer, hypothesis, [1100], 50)
test_dataset = NLIDataset(test_data, tokenizer, hypothesis, [1100], 50)

# train_dataset = train_dataset[:1]
# dev_dataset = dev_dataset[:1]
# test_dataset = test_dataset[:1] 

del train_data
del dev_data
del test_data
del hypothesis

def get_class_weights(dataset):
    nli_labels = [x['nli_label'] for x in dataset]

    span_labels = []
    for x in dataset:
        span_labels.extend(x['span_labels'].tolist())

    nli_weights = compute_class_weight('balanced', classes=np.unique(nli_labels), y=np.array(nli_labels))

    nli_weights = nli_weights.tolist()

    span_labels = [x for x in span_labels if x != -1]
    span_labels = np.array(span_labels)
    span_weight = np.sum(span_labels == 0) / np.sum(span_labels)

    return nli_weights, span_weight


nli_weights, span_weight = get_class_weights(train_dataset)


# print(nli_weights, span_weight)


# print("here")
# exit()

# Define model config
def model_init(trial=None):
    lambda_value = trial['lambda_'] if trial else 1.0
    return ContractNLI(ContractNLIConfig(
        nli_weights=nli_weights,
        span_weight=span_weight,
        lambda_=lambda_value,
        bert_model_name=cfg['model_name']
    ))

# Define Trainer subclass for Contract NLI
class ContractNLITrainer(Trainer):
    def __init__(self, *args, data_collator=None, **kwargs):
        super().__init__(*args, data_collator=data_collator, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        span_labels = inputs.pop('span_labels')
        nli_labels = inputs.pop('nli_label')
        
        outputs = model(**inputs)
        span_logits, nli_logits = outputs

        # Mask out padding tokens
        mask = span_labels != -1
        span_labels, span_logits = span_labels[mask].float(), span_logits[mask].float()

        # Compute losses
        span_loss = model.span_criterion(span_logits.view(-1), span_labels.view(-1)) if span_labels.numel() > 0 else 0
        nli_loss = model.nli_criterion(nli_logits, nli_labels)
        
        total_loss = span_loss + model.config.lambda_ * nli_loss
        return (total_loss, outputs) if return_outputs else total_loss

    @staticmethod
    def collate_fn(features):
        span_indices = [feature['span_indices'] for feature in features]
        max_len = max(len(indices) for indices in span_indices)
        
        span_indices = [torch.cat([indices, torch.zeros(max_len - len(indices), dtype=torch.long)]) for indices in span_indices]
        span_ids = [feature['data_for_metrics']['span_ids'] for feature in features]
        
        input_ids = torch.stack([f['input_ids'] for f in features])
        attention_mask = torch.stack([f['attention_mask'] for f in features])
        token_type_ids = torch.stack([f['token_type_ids'] for f in features])
        span_indices = torch.stack(span_indices)
        nli_labels = torch.stack([f['nli_label'] for f in features])
        span_labels = torch.cat([f['span_labels'] for f in features], dim=0)

        data_for_metrics = {
            'doc_id': torch.stack([f['data_for_metrics']['doc_id'] for f in features]),
            'hypothesis_id': torch.stack([f['data_for_metrics']['hypothesis_id'] for f in features]),
            'span_ids': torch.stack([torch.cat([ids, torch.full((max_len - len(ids),), -1)]) for ids in span_ids])
        }

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'span_indices': span_indices,
            'nli_label': nli_labels,
            'span_labels': span_labels,
            'data_for_metrics': data_for_metrics
        }

# Training configuration and arguments
training_args = TrainingArguments(
    output_dir=cfg['results_dir'],
    num_train_epochs=20,
    gradient_accumulation_steps=4,
    logging_strategy='steps',
    eval_steps=2,
    save_steps=2,
    logging_steps=2,
    evaluation_strategy='steps',
    save_strategy='steps',
    save_total_limit=2,
    load_best_model_at_end=True,
    fp16=True,
    auto_find_batch_size=True,
    label_names=['nli_label', 'span_labels', 'data_for_metrics'],
    report_to='none'
)

# Initialize Trainer with early stopping
trainer = ContractNLITrainer(
    model_init=model_init,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    data_collator=ContractNLITrainer.collate_fn,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.001)]
)

# Start training
trainer.train()

torch.save(trainer.model.state_dict(), 'model.pth')