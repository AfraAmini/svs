---
tags:
- generated_from_trainer
datasets:
- e2e_nlg_cleaned
metrics:
- accuracy
model-index:
- name: ckpts
  results:
  - task:
      name: Causal Language Modeling
      type: text-generation
    dataset:
      name: e2e_nlg_cleaned e2e_nlg_cleaned
      type: e2e_nlg_cleaned
      args: e2e_nlg_cleaned
    metrics:
    - name: Accuracy
      type: accuracy
      value: 0.5872205022359821
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# ckpts

This model is a fine-tuned version of [](https://huggingface.co/) on the e2e_nlg_cleaned e2e_nlg_cleaned dataset.
It achieves the following results on the evaluation set:
- Loss: 1.6906
- Accuracy: 0.5872

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 8
- eval_batch_size: 8
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 50.0

### Training results

| Training Loss | Epoch | Step  | Validation Loss | Accuracy |
|:-------------:|:-----:|:-----:|:---------------:|:--------:|
| 2.9736        | 1.25  | 500   | 2.1920          | 0.4790   |
| 2.0383        | 2.49  | 1000  | 1.8173          | 0.5633   |
| 1.7714        | 3.74  | 1500  | 1.7289          | 0.5790   |
| 1.6547        | 4.99  | 2000  | 1.6906          | 0.5872   |
| 1.561         | 6.23  | 2500  | 1.6983          | 0.5875   |
| 1.4924        | 7.48  | 3000  | 1.7095          | 0.5884   |
| 1.4334        | 8.73  | 3500  | 1.7194          | 0.5894   |
| 1.3732        | 9.98  | 4000  | 1.7407          | 0.5889   |
| 1.2915        | 11.22 | 4500  | 1.7971          | 0.5816   |
| 1.2221        | 12.47 | 5000  | 1.8376          | 0.5790   |
| 1.1512        | 13.72 | 5500  | 1.8742          | 0.5806   |
| 1.0772        | 14.96 | 6000  | 1.9357          | 0.5747   |
| 0.9817        | 16.21 | 6500  | 2.0160          | 0.5685   |
| 0.9019        | 17.46 | 7000  | 2.1121          | 0.5612   |
| 0.829         | 18.7  | 7500  | 2.1964          | 0.5583   |
| 0.7553        | 19.95 | 8000  | 2.2609          | 0.5587   |
| 0.6684        | 21.2  | 8500  | 2.3765          | 0.5549   |
| 0.5989        | 22.44 | 9000  | 2.4760          | 0.5498   |
| 0.544         | 23.69 | 9500  | 2.5571          | 0.5483   |
| 0.4861        | 24.94 | 10000 | 2.6245          | 0.5490   |
| 0.4231        | 26.18 | 10500 | 2.7350          | 0.5476   |
| 0.3757        | 27.43 | 11000 | 2.8070          | 0.5454   |
| 0.3378        | 28.68 | 11500 | 2.9025          | 0.5429   |
| 0.3024        | 29.93 | 12000 | 2.9465          | 0.5469   |
| 0.2637        | 31.17 | 12500 | 3.0513          | 0.5445   |
| 0.237         | 32.42 | 13000 | 3.1208          | 0.5440   |
| 0.2129        | 33.67 | 13500 | 3.1496          | 0.5431   |
| 0.1938        | 34.91 | 14000 | 3.2202          | 0.5446   |
| 0.1727        | 36.16 | 14500 | 3.2673          | 0.5465   |
| 0.1551        | 37.41 | 15000 | 3.3080          | 0.5433   |
| 0.1423        | 38.65 | 15500 | 3.3545          | 0.5426   |
| 0.13          | 39.9  | 16000 | 3.4009          | 0.5441   |
| 0.1179        | 41.15 | 16500 | 3.4429          | 0.5443   |
| 0.1075        | 42.39 | 17000 | 3.4842          | 0.5456   |
| 0.0995        | 43.64 | 17500 | 3.5246          | 0.5448   |
| 0.0926        | 44.89 | 18000 | 3.5363          | 0.5443   |
| 0.0855        | 46.13 | 18500 | 3.5715          | 0.5441   |
| 0.0791        | 47.38 | 19000 | 3.5858          | 0.5447   |
| 0.0754        | 48.63 | 19500 | 3.5998          | 0.5445   |
| 0.0718        | 49.88 | 20000 | 3.6020          | 0.5455   |


### Framework versions

- Transformers 4.21.0.dev0
- Pytorch 1.7.1+cu110
- Datasets 1.8.0
- Tokenizers 0.12.1
