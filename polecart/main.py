import torch
from argparse import ArgumentParser
import math
import datetime
from collections import deque
from params import (
    buffer_size,
    learning_rate,
    batch_size,
    policy_net_path,
)
from model import train_model, save_model, get_action, run_non_training
from utils import setup_env, reset_env, setup_policy_net, get_iterations
from reporting import Reporting
from epsilon import Epsilon


"""
An attempt at a DQN implementation for the CartPole environment.
Usage:
    poetry run python main.py --train
    poetry run python main.py --load <model_name> --train [--report]
    poetry run python main.py --load <model_name>

Will output a model file to saved_models/ with the current date and time if training is stopped,
as well as every 1000 iterations (configurable in params.py).

"""

def get_reward(degrees):
    return max(0, 1 - (abs(degrees) / 12))

def main(load=False, train=True, report=False):
    env = setup_env(train)
    state = reset_env(env)
    policy_net = setup_policy_net(env, load)
    # Initialize replay buffer
    replay_buffer = deque(maxlen=buffer_size)

    # Optimizer
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=learning_rate)
    # start epsilon at .01 if loading a model (assume pre-trained), otherwise start at 1
    epsilon = Epsilon(start=0.01 if load else 1, end=0.01, decay=0.9995)
    loss_fn = torch.nn.MSELoss()
    iterations = get_iterations(load)
    # Reporting metrics (plus iterations + epsilon - these shouldn't really live here, but it works.)
    reporting = Reporting(epsilon=epsilon, iterations=iterations, report=report)
    try:
        while True:
            if iterations > 100000000:
                break
            done = False
            reward = None
            reporting.reset_episode()
            while not done:
                if not train:
                    # Much simpler loop, with visualisation of the environment
                    state, reward, done = run_non_training(env, policy_net, state)
                else:
                    env.render()
                    action = get_action(env, policy_net, state, reporting)
                    result = env.step(action)
                    next_state, reward, done, _, _ = result
                    angle_rads = next_state[2]
                    degrees = math.degrees(angle_rads)
                    reward = get_reward(degrees)
                    replay_buffer.append((state, action, reward, next_state, done))

                    if len(replay_buffer) > batch_size * 10:
                        train_model(replay_buffer, policy_net, optimizer, loss_fn)
                    state = next_state
                reporting.round_len += 1

            if train:
                reporting.update_average_round_len()
                reporting.iterations += 1
                if reporting.should_save_model():
                    save_model(policy_net, f"saved_models/{policy_net_path}")
                reporting.report()
                reporting.epsilon.decay_epsilon()
            state = reset_env(env)

    except KeyboardInterrupt:
        if train:
            print("Training stopped")
            t = datetime.datetime.now()
            save_model(
                policy_net,
                f"saved_models/{t.strftime('%d-%m-%Y_%H-%M-%S')}_{iterations}_finalwithnewreward_{policy_net_path}",
            )
            env.close()


if __name__ == "__main__":
    argparse = ArgumentParser()
    argparse.add_argument("--load", type=str, required=False)
    argparse.add_argument("--train", action="store_true", default=False)
    argparse.add_argument("--report", action="store_true", default=False)
    args = argparse.parse_args()
    main(args.load, args.train, args.report)
