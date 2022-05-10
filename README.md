# PhraseRL

## Introduction

This repository contains implementation of PhraseRL, which is introduced in the following paper:

- [Phrase-Level Action Reinforcement Learning for Neural Dialog Response Generation](https://aclanthology.org/2021.findings-acl.446/) (Yamazaki and Aizawa, Findings of ACL 2021)

## Preparation

Use [poetry](https://github.com/python-poetry/poetry) to download dependencies.

```sh
# Install dependencies
poetry install
# Download MultiWOZ data
sh bin/fetch_data.sh
```

## Run

### Supervised Learning

The model first needs to be trained with supervised learning with the following commands:

```sh
# Train DISC model
poetry shell
python bin/train.py -c configs/disc.toml -s 0 -o outputs/disc
# Test
python bin/test.py -o outputs/disc -m best_model.pt
# Displaying outputs
python bin/display_model.py -o outputs/disc -n 10 -m best_model.pt
```

### Reinforcement Learning

To run additional training with reinforcement learning, execute the following commands:

```sh
# Train DISC model
poetry shell
python bin/policy.py -c outputs/disc/config.toml configs/rl.toml -s 0 -o outputs/disc-rl -m outputs/disc/best_model.pt
# Test
python bin/test.py -o outputs/disc-rl -m best_model.pt
# Displaying outputs
python bin/display_model.py -o outputs/disc-rl -n 10 -m best_model.pt
```
