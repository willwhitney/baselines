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
import ipdb

from gridworld import world


def wrap_train(env):
    return env


def train(args, num_frames, seed):
    from baselines.trpo_mpi.nosharing_cnn_policy import CnnPolicy, SmallCnnPolicy
    from baselines.ppo1.mlp_policy import MlpPolicy
    from baselines.trpo_mpi import trpo_mpi
    import baselines.common.tf_util as U
    rank = MPI.COMM_WORLD.Get_rank()
    sess = U.single_threaded_session()
    # tf_config = tf.ConfigProto(
    #     inter_op_parallelism_threads=1,
    #     intra_op_parallelism_threads=1)
    # sess = tf.Session(config=tf_config)
    sess.__enter__()
    if rank != 0:
        logger.set_level(logger.DISABLED)

    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)

    env = gym.make('GridWorld-v0')
    real_base_env = env.env
    real_base_env.setup(size=args.env_size, curriculum=args.curriculum,
                        walldeath=args.walldeath)
    env.observation_space = real_base_env.observation_space
    env.action_space = real_base_env.action_space

    test_env = gym.make('GridWorld-v0')
    real_test_env = test_env.env
    real_test_env.setup(size=args.env_size, curriculum=False,
                        walldeath=args.walldeath)
    test_env.observation_space = real_test_env.observation_space
    test_env.action_space = real_test_env.action_space

    def policy_fn(name, ob_space, ac_space):  # pylint: disable=W0613
        # ipdb.set_trace()
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

    trpo_mpi.learn(env, test_env, policy_fn,
                   timesteps_per_batch=1024, max_kl=args.max_kl, cg_iters=10, cg_damping=1e-3,
                   max_timesteps=num_timesteps, gamma=args.gamma, lam=args.lam,
                   vf_iters=args.vf_iters, vf_stepsize=args.vf_stepsize, entcoeff=0.0)

    # trpo_mpi.learn(env, policy_fn, timesteps_per_batch=512, max_kl=0.001, cg_iters=10, cg_damping=1e-3,
    #     max_timesteps=num_timesteps, gamma=0.98, lam=1.0, vf_iters=3, vf_stepsize=1e-4, entcoeff=0.00)
    env.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', default='default', help='experiment name')
    parser.add_argument('--env-size', default=8, choices=[8, 16, 28], type=int)
    parser.add_argument('--walldeath', default=0, type=int,
                        help='whether the agent "dies" when it hits a wall')
    parser.add_argument('--curriculum', default=0, type=int,
                        help='whether to train using a curriculum')
    parser.add_argument('--seed', default=0, help='RNG seed', type=int)
    parser.add_argument('--max-kl', help='KL divergence for policy updates',
                        default=0.01, type=float)
    parser.add_argument('--gamma', default=0.98, type=float, help='discount factor')
    parser.add_argument('--lam', default=0.98, type=float,
                        help='exponential decay of k-step estimator for GAE')
    parser.add_argument('--vf_iters', default=3, type=int,
                        help='steps to take on value function')
    parser.add_argument('--vf_stepsize', default=1e-4, type=float,
                        help='steps to take on value function')
    args = parser.parse_args()

    args.curriculum = (args.curriculum == 1)
    args.walldeath = (args.walldeath == 1)

    logdir = osp.join('results', args.name)
    os.makedirs(logdir, exist_ok=True)
    f = open(osp.join(logdir, 'results.json'), 'w')
    logger.Logger.DEFAULT = logger.Logger(
        dir=logdir,
        output_formats=[logger.HumanOutputFormat(sys.stdout),
                        logger.JSONOutputFormat(f)])
    logger.reset()
    train(args, num_frames=40e6, seed=args.seed)


if __name__ == "__main__":
    main()
