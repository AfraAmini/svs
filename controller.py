import logging
from typing import TypedDict

import pandas as pd


class ResultDict(TypedDict):
    sample: list
    ppl: list
    target: list
    target_str: list
    pred: list
    pred_str: list
    energy: list


class Controller:
    def __init__(self, c_model_info, args,
                 pad_token_id, target_dict):
        logging.info("HERE in main controlled body")
        self.device = args.device
        self.pad_token_id = pad_token_id
        self.save_dir = args.save_dir
        self.target_dict = target_dict
        self.controlled = args.controlled
        self.task = args.task

        self.g_ckpt = args.g_ckpt
        self.c_ckpt = args.c_ckpt
        self.result_dict: ResultDict = {"sample": [], "ppl": [], "energy": [], "pred": [],
                                        "target": [], "target_str": [], "pred_str": []}
        self.number_of_samples = args.number_of_samples
        logging.info("Right before going to initilizing g model")

        self.g_model, self.tokenizer = self.initialize_g_model(args.device)
        if self.controlled:
            self.c_model = self.initialize_c_model(c_model_info, args.device)
        else:
            self.c_model = None

    def initialize_g_model(self, device):
        pass

    def initialize_c_model(self, c_model_info, device):
        pass

    def predict_with_control(self, save_dir, name):
        pass

    def update_results_dict(self, sample, ppl, energy, target, pred):
        self.result_dict["sample"].append(sample)
        self.result_dict["ppl"].append(ppl)
        self.result_dict["energy"].append(energy)
        self.result_dict["target"].append(target)
        self.result_dict["pred"].append(pred)
        if self.target_dict:
            self.result_dict["target_str"].append(self.target_dict[target])
            self.result_dict["pred_str"].append(self.target_dict[pred])
        else:
            self.result_dict["target_str"].append("")
            self.result_dict["pred_str"].append("")

    def save_results_dict(self, save_dir, name):
        result_df = pd.DataFrame(self.result_dict)
        result_df.to_csv(save_dir + name + ".csv")
