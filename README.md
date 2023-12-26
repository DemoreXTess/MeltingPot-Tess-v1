# Tess-v1: Pure RL Solution for NeurIPS Melting Pot Contest

- [Melting Pot Contest (AICrowd)](https://www.aicrowd.com/challenges/meltingpot-challenge-2023)

Tess-v1 is a reinforcement learning solution developed for the NeurIPS Melting Pot Contest. The repository contains various training and model files, with `models/model_v4_cont.py` being applied in all the final experiments.
## Table of Contents

- [Introduction](#introduction)
- [Trainings](#trainings)
- [Prerequisites](#prerequisites)
- [License](#license)
- [Citation](#citation)

## Introduction

Tess-v1 is a reinforcement learning solution tailored for the challenges posed in the NeurIPS Melting Pot Contest. The repository encompasses various training and model files, with particular emphasis on `models/model_v4_cont.py`, which is employed in all the final experiments.

## Trainings

- **Harvest**: Trained using `train_multitask_harvest.py`
- **Clean Up**: Trained using `train_multitask.py`
- **Prisoners**: Trained using `train_multitask.py`
- **Territory Rooms**: Trained using `train-v6.py`

Note: For reproducibility, it is essential to configure the parameters in `configs.py`. The necessary configuration details will be shared in the next commit.

## Prerequisites

Before running Tess-v1, ensure you have the following prerequisites installed:

- [PyTorch](https://pytorch.org/) (version >= 2.0.1)
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
- Gym (version 0.21.0)
- OpenCV-Python
- NumPy
- WandB (Weights & Biases)
- MeltingPot Suite

You can install these dependencies using the following:

```bash
pip install torch>=2.0.1 stable-baselines3 gym==0.21.0 opencv-python numpy wandb dm-meltingpot
```
## License

This project is licensed under the [MIT License](https://github.com/utkuearas/MeltingPot-Tess-v1/blob/master/LICENSE)

## Tess-v1 RL Solution
```  
Version		: 1
Released 	: 2023-12-26
Author		:: Utku Erodoğanaras
Contacts 	: utkuearasbusiness@gmail.com
```

## Citation

Please site this repository in [this](https://github.com/utkuearas/MeltingPot-Tess-v1/blob/master/CITATION.cff) format.



