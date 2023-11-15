from typing import TypedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaForSequenceClassification

from svs.control.cvaluate import ClassificationMetric


class CModelConfig(TypedDict):
    ckpt: str
    pad_id: int
    hidden_dim: int
    vocab_size: int
    output_dim: int
    max_len: int


class RNNProbe(nn.Module):
    def __init__(self, c_model_info: CModelConfig, base_model, device):
        super(RNNProbe, self).__init__()
        self.hidden_dim = c_model_info['hidden_dim']

        self.rnn = nn.LSTM(self.hidden_dim, self.hidden_dim, num_layers=3,
                           bidirectional=True,
                           batch_first=True,
                           dropout=0.5)  # want it to be causal so we can learn all positions
        self.out_linear = nn.Linear(2 * self.hidden_dim, c_model_info['output_dim'])
        self.base_model = base_model
        self.device = device
        self.cm = ClassificationMetric()

    def forward(self, inputs: []):
        """
        inputs: token ids, batch x seq, right-padded with 0s
        lengths: lengths of inputs; batch
        """
        rnn_output, _ = self.rnn(inputs)
        return self.out_linear(rnn_output)

    def loss(self, inputs, lengths, labels, criterion): #TODO: there is no base model
        with torch.no_grad():
            hidden_states = \
                self.base_model(input_ids=inputs, output_hidden_states=True)['hidden_states'][
                    -1].to(self.device)

        self.train()
        scores = self.forward(hidden_states)
        out_dim = scores.shape[2]
        length_index = lengths.unsqueeze(1).repeat(1, 1, out_dim).permute(1, 0, 2)
        selected_score = scores.gather(1, length_index - 1).squeeze(1)
        loss = criterion(selected_score, labels)
        return loss, scores

    def energy(self, e, target):
        """Computes the classifier negative log probability for a given target class"""
        h = self.base_model(inputs_embeds=e[:, :-1, :],
                            output_hidden_states=True)['hidden_states'][-1]
        classifier_output = self.forward(h)[:, -1, :]
        log_probs = F.log_softmax(classifier_output, dim=-1)
        return -log_probs[:, target]

    def predict(self, e):
        h = self.base_model(inputs_embeds=e[:, :-1, :],
                            output_hidden_states=True)['hidden_states'][-1]
        classifier_output = self.forward(h)[:, -1, :]
        return torch.argmax(classifier_output, dim=-1)[0].item()


class RoBERTaEval(RobertaForSequenceClassification):
    """Class for the evaluator classifier"""
    def __init__(self, config):
        super().__init__(config)
        self.cm = ClassificationMetric()

    def loss(self, inputs, lengths, labels, criterion, base_model=None):
        output = self.forward(input_ids=inputs, labels=labels)
        return output.loss, output.logits
