import sys

sys.path.insert(0, '..')

from argparse import ArgumentParser

import numpy as np
import torch
from tqdm import tqdm as tq
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from control.models import RNNProbe
from controller import Controller
import json

HIDDEN_DIM = 768
BOS_ID = 764


class MuCoLa(Controller):
    def __init__(self, c_model_info, args, pad_token_id, target_dict=None):
        super().__init__(c_model_info, args, pad_token_id, target_dict)
        self.max_len = args.max_len

        self.steps = args.steps
        self.inner_steps = 100
        self.controlled = args.controlled
        self.step_size = args.step_size
        self.c_factor = args.c_factor

    def initialize_g_model(self, device):
        tokenizer = GPT2TokenizerFast.from_pretrained(self.g_ckpt)
        model = LangevinGPT2.from_pretrained(self.g_ckpt,
                                             pad_token_id=tokenizer.eos_token_id).to(
            self.device)
        return model, tokenizer

    def initialize_c_model(self, c_model_info, device):
        self.class_num = c_model_info['output_dim']
        ckpt = torch.load(self.c_ckpt, map_location=self.device)
        classifier = RNNProbe(c_model_info, self.g_model, self.device).to(self.device)
        classifier.load_state_dict(ckpt['state_dict'], strict=False)
        return classifier

    def initialize(self, context):
        input_ids = torch.randint(low=0, high=self.tokenizer.vocab_size,
                                  size=(1, self.max_len)).to(self.device)
        bos_tensor = context.to(self.device)
        input_ids = torch.cat([bos_tensor, input_ids], dim=-1)
        initial_e = self.g_model(input_ids, output_hidden_states=True)['hidden_states'][0]
        return initial_e, bos_tensor

    def project_embeds(self, e, context, context_len):
        with torch.no_grad():
            word_embeddings = self.g_model.transformer.wte.weight  # |V| x d
            vecs = e[0, context_len:, :]  # max_len x d

            projected_idx = torch.zeros((1, self.max_len)).long().to(self.device)
            for i in range(vecs.shape[0]):
                diff = word_embeddings - vecs[i]
                dists = torch.sum(diff * diff, dim=1)
                projected_idx[:, i] = torch.argmin(dists, dim=0)

            projected_idx = torch.cat([context, projected_idx], dim=-1)
            projected_e = \
            self.g_model(projected_idx, output_hidden_states=True)['hidden_states'][0]
        return projected_idx, projected_e

    def update_noise_variance(self, beta):
        if beta > 0.05:
            beta = beta * 0.93
            return beta
        return 0.05

    def early_stopping(self, last_e, e, tol):
        diff = last_e - e
        diff = torch.sum(diff * diff, dim=-1)
        if diff.mean() < 1e-5:
            tol -= 1
        else:
            tol = 10
        if tol == 0:
            return True, tol
        else:
            return False, tol

    def take_sample(self, e):
        sample = self.tokenizer.decode(e, skip_special_tokens=True)
        with torch.no_grad():
            ppl = self.g_model(input_ids=e, labels=e)[0]
            ppl = np.exp(ppl.detach().cpu().numpy())

        return sample, ppl

    def predict_with_control(self, save_dir, name, targets=None, contexts=None):
        if not self.controlled:
            targets = [0]  # dummy variable
        else:
            assert targets is not None  # for controlled sampling targets must be provided
        if contexts is None:
            contexts = [torch.tensor([BOS_ID]).unsqueeze(0)]
        for context in contexts:
            for target in targets:
                if self.controlled:
                    print(f"{'='*10}target: {self.target_dict[target]} {'='*10}")
                for _ in range(self.number_of_samples):
                    beta = self.step_size
                    initial_e, context_tensor = self.initialize(context)
                    context_len = context_tensor.shape[1]
                    e = torch.nn.Parameter(initial_e, requires_grad=True)
                    pbar = tq(range(self.steps))

                    tol = 10

                    with torch.enable_grad():
                        for i in pbar:
                            last_e = torch.clone(e)

                            optimizer = torch.optim.Adagrad([e], lr=self.step_size)
                            optimizer.zero_grad()

                            energy = self.g_model.energy_e_function(e, self.max_len,
                                                                    context_len).mean()

                            if self.controlled:
                                h = self.g_model(inputs_embeds=e,
                                                 output_hidden_states=True)['hidden_states'][
                                    -1]
                                energy = energy + self.c_factor * self.c_model.energy(h, target)

                            energy.backward()
                            optimizer.step()

                            epsilon = torch.normal(mean=0.0, std=1., size=e.data.size()).to(
                                self.device)
                            prev_e_value = (e.data + np.sqrt(
                                2 * self.step_size * beta) * epsilon).detach()  # v_n + stuff
                            projected_idx, projected_e = self.project_embeds(prev_e_value,
                                                                        context_tensor,
                                                                        context_len)
                            e = torch.nn.Parameter(projected_e)

                            beta = self.update_noise_variance(beta)

                            early_stop, tol = self.early_stopping(last_e, e, tol)

                            if early_stop:
                                break

                            if i % self.inner_steps == 0:
                                sample, ppl = self.take_sample(projected_idx[0])
                                print(sample, ppl)
                                print("=" * 100)

                            pbar.set_description("energy {}".format(energy.mean()),
                                                 refresh=True)
                            pbar.update()

                    sample = self.tokenizer.decode(projected_idx[0], skip_special_tokens=True)
                    ppl = np.exp(
                        self.g_model(input_ids=projected_idx, labels=projected_idx)[
                            0].detach().cpu().numpy())
                    if self.controlled:
                        pred = self.c_model.predict(e)
                    else:
                        pred = 0
                    energy = energy.mean().detach().cpu().numpy()
                    print(sample, ppl, energy)
                    self.update_results_dict(sample, ppl, energy, target, pred)
            self.save_results_dict(save_dir, name)


class LangevinGPT2(GPT2LMHeadModel):
    def energy_e_function(self, e, max_len, context_len=1):
        h = \
            self.forward(inputs_embeds=e[:, :-1, :], output_hidden_states=True)[
                'hidden_states'][
                -1][:, context_len - 1:, :]

        word_embeddings = self.transformer.wte.weight
        denom = h @ word_embeddings.T
        denom = torch.log(torch.sum(torch.exp(denom), -1))

        remained_e = (
                e[:, context_len:] - self.transformer.wpe.weight[
                                     context_len:max_len + context_len, :].unsqueeze(0))
        nom = torch.sum(remained_e * h, -1)
        loss = - nom + denom
        return loss

    def embedding_from_index(self, index, max_len):
        t_emd = self.transformer.wte.weight[index]
        p_embd = self.transformer.wpe.weight[:max_len + 1, :]
        return t_emd + p_embd


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=7, help='random seed')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--data_dir', type=str, default='datasets/')
    parser.add_argument('--dataname', type=str, default='e2e')

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--step_size', type=float, default=0.1)

    parser.add_argument('--max_len', type=int, default=15)
    parser.add_argument('--number_of_samples', type=int, default=100)
    parser.add_argument('--model', type=str, default="RNNProbe",
                        choices=["RNNProbe"])
    parser.add_argument('--task', type=str, default="food",
                        choices=["food", "sentiment"])

    parser.add_argument('--controlled', action='store_true', default=False)
    parser.add_argument('--c_factor', type=float, default=2., help='how to weight the control loss')

    parser.add_argument('--save_dir', type=str, required=True,
                        help='where to save the generations')
    parser.add_argument('--save_name', type=str, required=True,
                        help='name of the saved generation file')

    parser.add_argument('--g_ckpt', type=str, default="ckpts/gpt2-food")
    parser.add_argument('--c_ckpt', type=str, default="control/ckpts/food-probe")

    args = parser.parse_args()


    if args.task == "food":
        with open('ckpts/gpt2-food/model_config.json', 'r') as f:
            model_config = json.load(f)
        with open('ckpts/gpt2-food/map_dict.json', 'r') as f:
            map_dict = json.load(f)
    elif args.task == "sentiment":
        with open('ckpts/gpt2-food/model_config.json', 'r') as f:
            model_config = json.load(f)
        model_config['output_dim'] = 2
        model_config['hidden_dim'] = 1280
        map_dict = {0: "negative", 1: "positive"}



    if args.controlled:
        map_dict = {int(k): v for k, v in map_dict.items()}
        targets = list(range(model_config["output_dim"]))
    else:
        map_dict = None
        targets = [0]


    langevin = MuCoLa(model_config, args, model_config["pad_id"],
                                 target_dict=map_dict)
    contexts = None
    langevin.predict_with_control(args.save_dir, args.save_name, targets=targets,
                                  contexts=contexts)
    print(langevin.result_dict)
