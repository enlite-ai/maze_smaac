[![Maze-SMAAC Docker Image](https://github.com/enlite-ai/maze_smaac/actions/workflows/github-ci.yml/badge.svg)](https://github.com/enlite-ai/maze_smaac/actions/workflows/github-ci.yml)

# Semi-Markov Afterstate Actor-Critic (SMAAC) with Maze

![Example for trained agent in action](https://cdn-images-1.medium.com/max/873/1*qYrrwQLdkf21voqSXGQ7zQ.gif)


The ["Learning to run a power network" (L2RPN)](https://l2rpn.chalearn.org/) challenge is a series of competitions proposed by [Kelly at al. (2020)](https://arxiv.org/pdf/2003.07339.pdf) with the aim to test the potential of reinforcement learning to control electrical power transmission. The challenge is motivated by the fact that existing methods are not adequate for real-time network operations on short temporal horizons in a reasonable compute time. Also, power networks are facing a steadily growing share of renewable energy, requiring faster responses. This raises the need for highly [robust](https://competitions.codalab.org/competitions/25426) and [adaptive](https://competitions.codalab.org/competitions/25427) power grid controllers.
In 2020, [one such competition](https://competitions.codalab.org/competitions/24902) was run at the IEEE World Congress on Computational Intelligence (WCCI) 2020. The winners have published their novel approach of combining a [Semi-MDP with an after-state representation](https://openreview.net/pdf?id=LmUJqB1Cz8) at ICLR 2021 and made their [implementation](https://github.com/KAIST-AILab/SMAAC) publicly available. The [latest iteration of the L2RPN challenge](https://icaps21.icaps-conference.org/Competitions/) poses a welcome opportunity to introduce our [RL framework Maze](https://github.com/enlite-ai/maze) and to replicate the winning approach with it.

This poses a welcome opportunity to introduce our RL framework Maze by replicating the SMAAC approach with it. This repository contains all necessary code and instructions for this. For a more extensive wrap up you can also check out our [accompanying blog post](https://medium.com/).


## Maze

![Banner](https://github.com/enlite-ai/maze/raw/main/docs/source/logos/main_logo.png)

[Maze](https://github.com/enlite-ai/maze) is an application-oriented deep reinforcement learning (RL) framework, addressing real-world decision problems.
Our vision is to cover the complete development life-cycle of RL applications, ranging from simulation engineering to agent development, training and deployment.
  
If you encounter a bug, miss a feature or have a question that the [documentation](https://maze-rl.readthedocs.io/) doesn't answer: We are happy to assist you! Report an [issue](https://github.com/enlite-ai/maze/issues) or start a discussion on [GitHub](https://github.com/enlite-ai/maze/discussions) or [StackOverflow](https://stackoverflow.com/questions/tagged/maze-rl).


## Installation 

### As Conda environment

Install all dependencies:
```shell
conda env create -f environment.yml
conda activate maze_smaac
```

Install [lightsim2grid](https://github.com/BDonnot/lightsim2grid), a fast backend for [Grid2Op](https://github.com/rte-france/Grid2Op):
```shell
chmod +x install_lightsim2grid.sh
./install_lightsim2grid.sh
```

Optional: Install this repository to include it in your Python path with
```shell
pip install -e .
```

### As Docker image

Execute   
```shell
docker buildx build -t enliteai/maze_smaac --build-arg MAZE_CORE_ENV=enliteai/maze:latest -f docker/maze_smaac.dockerfile .
```
to locally build a Docker image. Alternatively pull it with:
```shell
docker pull enliteai/maze_smaac:latest /bin/bash
```
Start a container with:
```shell
docker run -it enliteai/maze_smaac:latest /bin/bash
```

## Data Download

You need the chronics data from the official SMAAC repo. The [training script](#via-python-api) downloads and unpacks the required data automatically if it's not available in `maze_smaac/data`. The data download can also be started explicitly with
```shell
python scripts/data_acquisition.py
```

Alternatively you can manually download the data from [here](https://drive.google.com/file/d/15oW1Wq7d6cu6EFS2P7A0cRhyv8u_UqWA/view?usp=sharing). The extracted data folder should replace `maze_smaac/data`.

## Test Installation

> **_NOTE:_**  If you haven't run `pip install -e .`, you need to prefix all CLI commands with `PYTHONPATH='.'`.

If everything is installed correctly, this command should execute successfully:
```shell
maze-run -cn conf_train env=maze_smaac model=maze_smaac_nets wrappers=maze_smaac_debug +experiment=sac_dev

```

# Training

## Via CLI 

Start training in train mode: 
```shell
maze-run -cn conf_train env=maze_smaac model=maze_smaac_nets wrappers=maze_smaac_train +experiment=sac_train
```

Start training in debug mode: 
```shell
maze-run -cn conf_train env=maze_smaac model=maze_smaac_nets wrappers=maze_smaac_debug +experiment=sac_dev
```

Perform rollout of trained policy:
```shell
maze-run policy=torch_policy env=maze_smaac model=maze_smaac_nets wrappers=maze_smaac_rollout runner=evaluation input_dir=EXPERIMENT_LOGDIR
``` 

## Via Python API 

For invoking training, evaluation and rollout in Python, run the Python script utilizing Maze' Python API:  
```shell
python scripts/training.py
```
We encourage you to use the snippets in `training.py` as a starting point to customize the training configuration and write your own scripts.   
