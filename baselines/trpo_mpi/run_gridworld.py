#!/usr/bin/env python
from mpi4py import MPI
from baselines.common import set_global_seeds
import os.path as osp
import os
import gym
import logging
from baselines import logger
from baselines import bench
import sys
import pdb

from gridworld import world


def wrap_train(env):
    return env


def train(env_id, num_frames, seed):
    from baselines.trpo_mpi.nosharing_cnn_policy import CnnPolicy, SmallCnnPolicy
    from baselines.ppo1.mlp_policy import MlpPolicy
    from baselines.trpo_mpi import trpo_mpi
    import baselines.common.tf_util as U
    rank = MPI.COMM_WORLD.Get_rank()
    sess = U.single_threaded_session()
    sess.__enter__()
    if rank != 0:
        logger.set_level(logger.DISABLED)

    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)

    env = gym.make('GridWorld-v0')

    def policy_fn(name, ob_space, ac_space):  # pylint: disable=W0613
        return SmallCnnPolicy(name=name,
                              ob_space=env.observation_space,
                              ac_space=env.action_space)
        # return MlpPolicy(name=name, ob_space=env.observation_space, ac_space=env.action_space,
        #                  hid_size=32, num_hid_layers=2)

    env = bench.Monitor(env, logger.get_dir() and
                        osp.join(logger.get_dir(), "%i.monitor.json" % rank))
    env.seed(workerseed)
    gym.logger.setLevel(logging.WARN)

    env = wrap_train(env)
    num_timesteps = int(num_frames)
    env.seed(workerseed)

    trpo_mpi.learn(env, policy_fn,
                   timesteps_per_batch=512, max_kl=0.001, cg_iters=10, cg_damping=1e-3,
                   max_timesteps=num_timesteps, gamma=0.9, lam=0.99, vf_iters=3,
                   vf_stepsize=1e-4, entcoeff=0.01)
    # trpo_mpi.learn(env, policy_fn, timesteps_per_batch=512, max_kl=0.001, cg_iters=10, cg_damping=1e-3,
    #     max_timesteps=num_timesteps, gamma=0.98, lam=1.0, vf_iters=3, vf_stepsize=1e-4, entcoeff=0.00)
    env.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID',
                        default='PongNoFrameskip-v4')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--name', help='experiment name', default='default')
    args = parser.parse_args()

    logdir = osp.join('results', args.name)
    os.makedirs(logdir, exist_ok=True)
    f = open(osp.join(logdir, 'results.json'), 'w')
    logger.Logger.DEFAULT = logger.Logger(
        dir=logdir,
        output_formats=[logger.HumanOutputFormat(sys.stdout),
                        logger.JSONOutputFormat(f)])
    logger.reset()
    train(args.env, num_frames=40e6, seed=args.seed)


if __name__ == "__main__":
    main()
