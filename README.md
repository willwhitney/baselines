# Running gridworld experiments with TRPO

command: 
`mpirun -np {cores} python -m baselines.trpo_mpi.run_gridworld {flags}`

grid:

```
curriculum: [0, 1], # advance to step k+1 when reward is >= 1 - k/35
walldeath: [0, 1], # episode ends if the agent runs into a wall
env-size: [8],
max-kl: [0.01, 0.001],
lam: [1.0, 0.98],
seed: [0, 1, 2],
```


<img src="data/logo.jpg" width=25% align="right" />

# Baselines

OpenAI Baselines is a set of high-quality implementations of reinforcement learning algorithms.

These algorithms will make it easier for the research community to replicate, refine, and identify new ideas, and will create good baselines to build research on top of. Our DQN implementation and its variants are roughly on par with the scores in published papers. We expect they will be used as a base around which new ideas can be added, and as a tool for comparing a new approach against existing ones. 

You can install it by typing:

```bash
git clone https://github.com/openai/baselines.git
cd baselines
pip install -e .
```

- [A2C](baselines/a2c)
- [ACKTR](baselines/acktr)
- [DDPG](baselines/ddpg)
- [DQN](baselines/deepq)
- [PPO](baselines/ppo1)
- [TRPO](baselines/trpo_mpi)
