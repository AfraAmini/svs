import random
from collections import defaultdict

import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src import logger


class Dataset(torch.utils.data.Dataset):
    def __init__(self, args, tokenizer_str, split):
        logger.info('loading data')
        random.seed(args.seed)
        self.batch_size = args.batch_size
        self.split = split

        self.data_dir = args.data_dir
        self.dataname = args.dataname
        self.max_len = args.max_len

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_str)
        self.pad_id = 0

    def collate(self, batch):
        inputs = [b['input_ids'] for b in batch]
        lengths = torch.LongTensor([len(b['input_ids']) for b in batch])
        lengths = torch.clamp(lengths, max=512)

        if self.max_len != -1:
            max_length = self.max_len
        else:
            max_length = min(lengths.max(), 512)

        for i in range(len(inputs)):
            if len(inputs[i]) < max_length:
                inputs[i] = torch.cat(
                    [inputs[i],
                     torch.zeros(max_length - len(inputs[i])).long() + self.pad_id],
                    dim=0)  # 0 is fine as pad since it's masked out
            else:
                inputs[i] = inputs[i][:max_length]
        inputs = torch.stack(inputs, dim=0)

        labels = [b['label'] for b in batch]
        labels = torch.LongTensor(labels)

        return inputs, lengths, labels


class SST2Dataset(Dataset):
    def __init__(self, args, tokenizer_str, pad_token, split):
        super().__init__(args, tokenizer_str, split)
        self.tokenizer.add_special_tokens({'pad_token': pad_token})
        self.pad_id = 0
        print(self.data_dir)
        print(self.dataname)
        self.df = pd.read_csv(self.data_dir + self.dataname + "/" + split + ".csv")
        self.class_num = 2
        self.le_name_mapping = {0: "negative", 1: "positive"}

        logger.info('done loading data')
        logger.info('split {} size: {}'.format(split, len(self.df)))

    def __getitem__(self, index):
        raw_sentence = self.df.iloc[index]["sentence"]
        label = int(self.df.iloc[index]["label"])
        encoded = self.tokenizer.encode(raw_sentence, return_tensors='pt')[0]

        return {'input_ids': encoded, 'length': len(encoded), 'label': label}

    def __len__(self):
        return len(self.df)


class FoodDataset(Dataset):
    def __init__(self, args, label_col, tokenizer_str, pad_token, split):
        super().__init__(args, tokenizer_str, split)

        self.label_col = label_col

        self.tokenizer.add_special_tokens({'pad_token': pad_token})
        self.pad_id = 0
        # self.pad_id = self.tokenizer.encode(pad_token)[0]
        self.vocab = defaultdict(lambda: 0)
        self.splits = []
        self.split_labels = []

        self.labels_vocab = set()
        self.label_encoder = LabelEncoder()

        with open(self.data_dir + self.dataname + "_data/src1_" + split + ".txt") as f:
            for line in f:
                text = line.split("||")[1].strip()
                properties = line.split("||")[0].strip()
                properties = {s.split(":")[0].strip(): s.split(":")[1].strip() for s in
                              properties.split("|")}
                if self.label_col not in properties:
                    continue

                for word in text.strip().split(' '):
                    self.vocab[word] += 1
                self.split_labels.append(properties[self.label_col])
                self.labels_vocab.add(properties[self.label_col])
                self.splits.append(text)

        self.class_num = len(self.labels_vocab)
        self.label_encoder.fit(list(self.labels_vocab))
        self.le_name_mapping = dict(
            zip(self.label_encoder.transform(self.label_encoder.classes_),
                self.label_encoder.classes_))
        logger.info(self.le_name_mapping)
        self.split_labels = self.label_encoder.transform(self.split_labels)
        self.splits = tuple(zip(self.splits, self.split_labels))

        logger.info('done loading data')
        logger.info('split {} size: {}'.format(split, len(self.splits)))

    def __getitem__(self, index):
        raw_sentence = self.splits[index][0]
        label = self.splits[index][1]
        encoded = self.tokenizer.encode(raw_sentence, return_tensors='pt')[0]

        return {'input_ids': encoded, 'length': len(encoded), 'label': label}

    def __len__(self):
        return len(self.splits)


def loader(args, label_col, tokenizer_str, pad_token, split):
    dataset = None
    if args.dataname == "e2e":
        dataset = FoodDataset(args, label_col, tokenizer_str, pad_token, split)
    elif args.dataname == "sst2":
        dataset = SST2Dataset(args, tokenizer_str, pad_token, split)

    return DataLoader(dataset,
                      batch_size=args.batch_size,
                      collate_fn=dataset.collate,
                      shuffle=True)
