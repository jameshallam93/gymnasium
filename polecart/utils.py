import gym
from model import PolicyNetwork
import numpy as np
import torch


def setup_env(train):
    kwargs = {}
    if not train:
        kwargs["render_mode"] = "human"
    env = gym.make("CartPole-v1", **kwargs)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=10000)
    print("Env created")
    return env


def reset_env(env):
    state, _ = env.reset()
    state = np.array(state)
    return state


def setup_policy_net(env, load):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    policy_net = PolicyNetwork(state_dim, action_dim)
    if load:
        policy_net.load_state_dict(torch.load(f"saved_models/{load}"))
    return policy_net


def get_iterations(load):
    if load:
        return int(load.split("_")[2])
    return 0
