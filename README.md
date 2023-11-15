# Data
To build the vocabulary of models, train the probing classifiers, and evaluate them make sure that
data is downloaded in `datasets` directory (or change `--data_dir` argument in the scripts).
# Toy Model
To reproduce the plots and results on the toy model, run `ToyExperiment.ipynb` notebook.
# Training the Classifiers
There are two types of classifiers that are used in this paper. We explain what commands to use 
to train each below. Code for training, evaluation, and model architectures are in `control` directory.
1. probing classifiers that are used to guide the generating process. We train bidirectional LSTM layers on tor of GPT-2 models, with the command below:
```bash
python control/train.py --model RNNProbe --task food --base_model_str gpt2 --save_dir [CKPT_DIR] --save_name [CKPT_NAME]
```
2. evaluator classifier that are used to evaluate the quality of the generated text. We finetune a roberta model, with the command below:
```bash 
python control/train.py --model EVAL --task food --base_model_str roberta-base --save_dir [CKPT_DIR] --save_name [CKPT_NAME]
```
Use `--task food` for topic control and `--task sentiment` for sentiment control.

# Generating Text
This repository includes re-implementations of MuCoLa and implementation of SVS.
## MuCoLa
For generating text from the LM without enforcing any control:
```bash 
python mucola.py --save_dir [SAVE_DIR] --save_name [SAVE_NAME] --step_size 0.1 --steps 500
```
For generating text from LM with topic control (`food` dataset):
```bash 
python mucola.py --task food  --save_dir [SAVE_DIR] --save_name [SAVE_NAME] --step_size 0.1 --c_factor 2. --steps 500 --controlled
```
For generating text from LM with sentiment control:
```bash 
python mucola.py --g_ckpt gpt2-large --c_ckpt control/ckpts/sst2-probe-large --task sentiment --save_dir [SAVE_DIR] --save_name [SAVE_NAME] --step_size 0.6 --c_factor 1.5 --steps 500 --controlled
```
## SVS