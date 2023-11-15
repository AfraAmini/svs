import random

import torch
from sklearn.metrics import classification_report


class ClassificationMetric(object):
    """Class for computing the classification metrics used for training the classifiers"""
    def __init__(self):
        random.seed(0)
        self.predicted_labels = []
        self.gold_labels = []
        self.do_evalb = False

    def reset(self):
        self.predicted_labels = []
        self.gold_labels = []

    def update(self, scores, lengths, labels):
        if len(scores.size()) > 2:  # we have scores over all word tokens
            indices = lengths - 1
            indices = indices[:, None, None].expand(scores.shape)
            filtered_scores = torch.gather(scores, 1, indices)[:, 0, :].squeeze(1)
        else:  # we only have sequence scores
            filtered_scores = scores

        self.predicted_labels += torch.argmax(filtered_scores, dim=-1).tolist()
        self.gold_labels += labels.tolist()

    def get_report(self):
        cr = classification_report(self.gold_labels, self.predicted_labels, output_dict=True)
        print(cr)
        return cr['weighted avg']
