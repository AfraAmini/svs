import sys

sys.path.insert(0, '..')

import os
import random
import time
from argparse import ArgumentParser

from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import wandb
from tqdm import tqdm
from transformers import GPT2LMHeadModel

from src import logger
from src.consts import HIDDEN_DIM, PAD_TOKEN, EOS_TOKEN_ID
from src.control.models import CModelConfig, RNNProbe, RoBERTaEval
from src.data import loader
from src.utils import save_checkpoint, num_params


def initialize_dataloaders():
    if args.task == "food":
        args.dataname = "e2e"
    else:
        args.dataname = args.task
    train_dataloader = loader(args, args.task, args.base_model_str, PAD_TOKEN, "train")
    pad_id = train_dataloader.dataset.pad_id
    class_num = train_dataloader.dataset.class_num
    vocab_size = train_dataloader.dataset.tokenizer.vocab_size

    valid_dataloader = loader(args, args.task, args.base_model_str, PAD_TOKEN, "valid")
    test_dataloader = loader(args, args.task, args.base_model_str, PAD_TOKEN, "test")

    return pad_id, class_num, vocab_size, train_dataloader, valid_dataloader, test_dataloader


def initialize_model(pad_id=0, vocab_size=0, class_num=0):
    model_config: CModelConfig = {"ckpt": args.ckpt,
                                  "pad_id": pad_id,
                                  "hidden_dim": args.hidden_dim,
                                  "vocab_size": vocab_size,
                                  "output_dim": class_num,
                                  "max_len": args.max_len}
    if args.model == "RNNProbe":  # probing classifier
        model = RNNProbe(model_config, base_model, args.device)
    elif args.model == "EVAL":  # evaluator classifier
        model = RoBERTaEval.from_pretrained(args.base_model_str, num_labels=class_num)
    else:
        logger.error("Model is not valid")
        return

    model = model.to(args.device)
    return model


def train_classifier(dataloader, valid_dataloader, pad_id, vocab_size, class_num):
    os.makedirs(args.save_dir, exist_ok=True)
    model = initialize_model(pad_id=pad_id, vocab_size=vocab_size, class_num=class_num)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val_metric = 1e8  # lower is better
    logger.info('num params', num_params(model))
    criterion = nn.CrossEntropyLoss().to(args.device)

    validate(model, valid_dataloader, criterion)

    if args.evaluate:
        validate(model, valid_dataloader, criterion)
        return

    step = 0
    tol = 10  # patience tolerance for early stopping

    for epoch in range(args.epochs):
        logger.info("TRAINING: Epoch {} at {}".format(epoch, time.ctime()))

        for batch_num, batch in enumerate(tqdm(dataloader, total=len(dataloader))):
            batch = [tensor.to(args.device) for tensor in batch]
            inputs, lengths, labels = batch

            loss, socres = model.loss(inputs, lengths, labels, criterion,
                                      base_model=base_model)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logger.logkv("train_loss", loss.detach())
            logger.logkv("step", step)
            if batch_num % args.train_print_freq == 0:
                logger.dumpkvs()

            step += 1

        if epoch % args.validation_freq == 0:
            logger.info("VALIDATION: Epoch {} at {}".format(epoch, time.ctime()))
            metric = validate(model, valid_dataloader, criterion)

            if not args.debug:
                if metric < best_val_metric:
                    tol = 10
                    logger.info('new best val metric', metric)
                    best_val_metric = metric
                    save_checkpoint({
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'best_metric': best_val_metric,
                        'optimizer': optimizer.state_dict(),
                        'data_start_index': step,
                        'args': args
                    }, os.path.join(args.save_dir, args.save_name))
                else:
                    tol -= 1
                if tol == 0:
                    return


def validate(model, loader, criterion, split="valid"):
    model.eval()
    random.seed(0)
    all_loss = []
    cm = model.cm
    cm.reset()
    with torch.no_grad():
        for batch_num, batch in enumerate(tqdm(loader, total=len(loader))):
            batch = [tensor.to(args.device) for tensor in batch]
            inputs, lengths, labels = batch
            loss, scores = model.loss(inputs, lengths, labels, criterion,
                                      base_model=base_model)

            logger.logkv_mean(f"{split}_loss", loss.detach())
            all_loss.append(loss.item())
            cm.update(scores, lengths, labels)

    avg_loss = np.mean(np.asarray(all_loss))
    logger.logkv_mean(f"{split}_loss", loss.detach())

    logger.logkvs(cm.get_report())
    logger.info(avg_loss)
    logger.dumpkvs()
    return avg_loss


def test_classifier(pad_id, vocab_size, class_num):
    model = initialize_model(pad_id=pad_id, vocab_size=vocab_size, class_num=class_num)
    model.load_state_dict(
        torch.load(args.ckpt, map_location=args.device)['state_dict'], strict=False)
    criterion = nn.CrossEntropyLoss().to(args.device)
    metric = validate(model, test_dataloader, criterion)
    print(metric)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=7, help='random seed')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--data_dir', type=str, default='../datasets/')
    parser.add_argument('--dataname', type=str, default='e2e')

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--hidden_dim', type=int, default=HIDDEN_DIM)
    parser.add_argument('--lr', type=float, default=1e-4, help='Adam learning rate')

    parser.add_argument('--max_len', type=int, default=-1)
    parser.add_argument('--model', type=str, default="EVAL",
                        choices=["RNNProbe", "EVAL"])

    parser.add_argument('--task', type=str, default="sst2",
                        choices=["food", "sst2"])

    parser.add_argument('--save_dir', type=str, required=True,
                        help='where to save ckpts')
    parser.add_argument('--save_name', type=str, required=True,
                        help='name of the saved checkpoint')
    parser.add_argument('--ckpt', type=str, default="checkpoints/probe")

    parser.add_argument('--train_print_freq', type=int, default=10,
                        help='how often to print metrics (every X batches)')
    parser.add_argument('--validation_freq', type=int, default=1,
                        help='validate every X epochs')

    parser.add_argument('--evaluate', action='store_true', default=False)
    parser.add_argument('--base_model_str', type=str, default="roberta-base",
                        help='for the probing classifier this is the checkpoint of the language '
                             'model that we probe. For the evaluator classifier this should be '
                             'roberta checkpoint to reproduce the paper reuslts.')
    parser.add_argument('--test', action='store_true', default=False)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    logger.configure(wandb=args.verbose)

    if args.verbose:
        wandb.init(
            project="cg-train-classifier",
            name="{} {} [{}]".format(args.task, args.model, datetime.now()),
        )

    if not args.model == "EVAL":
        base_model = GPT2LMHeadModel.from_pretrained(args.base_model_str,
                                                     pad_token_id=EOS_TOKEN_ID).to(
            args.device)
    else:
        base_model = None

    pad_id, class_num, vocab_size, train_dataloader, valid_dataloader, test_dataloader = \
        initialize_dataloaders()
    if not args.test:
        train_classifier(train_dataloader, valid_dataloader, pad_id, vocab_size,
                         class_num)
    else:
        test_classifier(pad_id, vocab_size, class_num)
