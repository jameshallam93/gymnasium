

import gym
import torch
import numpy as np
from argparse import ArgumentParser

import random
import datetime
from collections import deque
from params import buffer_size, learning_rate, batch_size, save_interval, policy_net_path
from model import PolicyNetwork, train_model, save_model

# Create the CartPole environment
def main(load=False, train=True):
    # TODO: Refactor this mess
    kwargs = {}
    if not train:
        kwargs["render_mode"] = "human"
    env = gym.make('CartPole-v1',**kwargs)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=10000)
    print("Env created")
    state, _ = env.reset()
    state = np.array(state) 
    # print(state)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    policy_net = PolicyNetwork(state_dim, action_dim)
    if load:
        policy_net.load_state_dict(torch.load(f"saved_models/{load}"))
    # Initialize replay buffer
    replay_buffer = deque(maxlen=buffer_size)

    # Optimizer
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=learning_rate)
    epsilon = 1.0  # Start with full exploration
    epsilon_min = 0.01
    epsilon_decay = 0.9995  # Decrease exploration over time
    # assume the model is already trained past the point of needing to explore
    if load:
        epsilon = 0.01
    # Loss function
    loss_fn = torch.nn.MSELoss()
    iterations = 0
    if load:
        iterations = int(load.split("_")[2])
    average_round_len = 0
    l_a = 0
    r_a = 0
    try:
        while True:
            if iterations > 100000000:
                break
            done = False
            reward = None
            round_len = 0
            rand = 0
            network = 0

            while not done:
                if not train:
                    # Preprocess the state as needed
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Convert to tensor and add batch dimension

                    # Use the loaded model to select an action
                    action = policy_net(state_tensor).argmax().item()

                    # Take the selected action in the environment
                    next_state, reward, done, _, _ = env.step(action)

                    # Update the current state
                    state = next_state
                    env.render()  # Optionally, render the environment to visualize the agent's behavior

                else:
                    env.render()
                    # use a random action with probability epsilon for exploration
                    if random.random() < epsilon:
                        action = env.action_space.sample()  # Random action for exploration
                        rand += 1
                    else:
                        action = policy_net(torch.FloatTensor(state).unsqueeze(0)).argmax().item()
                        # print("Action:", action)
                        network += 1
                        if action == 0:
                            l_a += 1
                        else:
                            r_a += 1
                    result = env.step(action)
                    
                    # x = math.degrees(result[0][2])
                    # print("Pole angle:", x)
                    next_state, reward, done, _, _ = result  
                    replay_buffer.append((state, action, reward, next_state, done))
                    if len(replay_buffer) > batch_size * 10:
                        train_model(replay_buffer, policy_net, optimizer, loss_fn)
                    # time.sleep(0.5)
                    state = next_state
                    round_len += 1
                # time.sleep(1)
            if average_round_len == 0:
                average_round_len = round_len
            else:
                average_round_len = (average_round_len + ((round_len - average_round_len) / (iterations + 1)))
            iterations += 1
            if iterations > 10000 and iterations % save_interval == 0:
                if train:
                    save_model(policy_net, f"saved_models/{policy_net_path}")
            if iterations % 100 == 0:

                print("Average round length:", average_round_len)
                print("Epsilon:", epsilon)
                print("Iterations:", iterations)
                if train:
                    print("Left actions:", l_a)
                    print("Right actions:", r_a)
                    print("Random actions:", rand)
                    print("Network actions:", network)
                average_round_len = 0
                r_a = 0
                l_a = 0

            epsilon *= epsilon_decay
            if epsilon < epsilon_min:
                epsilon = epsilon_min
            state, _ = env.reset()
            state = np.array(state)
    except KeyboardInterrupt:
        print("Training stopped")
        t = datetime.datetime.now()
        save_model(policy_net, f"saved_models/{t.strftime('%d-%m-%Y_%H-%M-%S')}_{iterations}_final_{policy_net_path}")
        env.close()

if __name__ == "__main__":
    argparse = ArgumentParser()
    argparse.add_argument("--load", type=str, required=False)
    argparse.add_argument("--train", action="store_true", default=False)
    args = argparse.parse_args()
    main(args.load, args.train)