# contractNliModel.py

import os
import torch
import numpy as np
from transformers import PreTrainedModel, PretrainedConfig, AutoModel
from torch import nn
from utils import get_labels
from prepare_dataset import cfg


class ContractNLIConfig(PretrainedConfig):

    # def __init__(self, lambda_ = 1, bert_model_name = cfg['model_name'], num_labels = len(get_labels()), ignore_span_label = 2, nli_weights = nli_weights, span_weight = span_weight, **kwargs):
    def __init__(self, nli_weights = [1, 1, 1], span_weight = 1, lambda_ = 1, bert_model_name = cfg['model_name'], num_labels = len(get_labels()), ignore_span_label = 2, **kwargs):
        super().__init__(**kwargs)
        self.bert_model_name = bert_model_name
        self.num_labels = num_labels
        self.lambda_ = lambda_
        self.ignore_span_label = ignore_span_label
        self.nli_weights = nli_weights
        self.span_weight = span_weight

class ContractNLI(PreTrainedModel):

    config_class = ContractNLIConfig

    def __init__(self, config):
        super().__init__(config)
        self.bert = AutoModel.from_pretrained(config.bert_model_name)
        self.bert.resize_token_embeddings(self.bert.config.vocab_size + 1, pad_to_multiple_of=8)
        self.bert.eval()
        for param in self.bert.parameters():
            param.requires_grad = False

        self.embedding_dim = self.bert.config.hidden_size
        self.num_labels = config.num_labels
        self.lambda_ = config.lambda_
        self.nli_criterion = nn.CrossEntropyLoss(weight=torch.tensor(self.config.nli_weights, dtype=torch.float32))
        self.span_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.config.span_weight, dtype=torch.float32))
        self.span_classifier = nn.Sequential(
            nn.Linear(self.embedding_dim * 4, self.embedding_dim * 4),
            nn.ReLU(),
            nn.Linear(self.embedding_dim * 4, self.embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(self.embedding_dim * 2, 1)
        )
        self.nli_classifier = nn.Sequential(
            nn.Linear(self.embedding_dim * 4, self.embedding_dim * 4),
            nn.ReLU(),
            nn.Linear(self.embedding_dim * 4, self.embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(self.embedding_dim * 2, self.num_labels)
        )
        # initialize weights
        self.init_weights()


    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # use the same initialization as bert
            module.weight.data.normal_(mean=0.0, std=self.bert.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input_ids, attention_mask, token_type_ids, span_indices):
        outputs = self.bert(input_ids, attention_mask, token_type_ids, output_hidden_states=True).hidden_states[-4:]
        outputs = torch.stack(outputs, dim=0)
        outputs = outputs.permute([1, 2, 0, 3])
        outputs = outputs.reshape([outputs.shape[0], outputs.shape[1], -1])
        gather = torch.gather(outputs, 1, span_indices.unsqueeze(2).expand(-1, -1, outputs.shape[-1]))
        masked_gather = gather[span_indices != 0]
        span_logits = self.span_classifier(masked_gather)
        nli_logits = self.nli_classifier(outputs[:, 0, :])
        return span_logits, nli_logits
