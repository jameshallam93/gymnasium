

import gym
import torch
import numpy as np

import random
from collections import deque
from params import buffer_size, learning_rate, batch_size
from model import PolicyNetwork, train_model

# Create the CartPole environment
def main():
    env = gym.make('CartPole-v1', render_mode="human")
    env = gym.wrappers.TimeLimit(env, max_episode_steps=10000)
    print("Env created")
    # print(env)
    state, _ = env.reset()
    state = np.array(state) 
    print(state)
    # return
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    print(action_dim)
    policy_net = PolicyNetwork(state_dim, action_dim)
     
    # Initialize replay buffer
    replay_buffer = deque(maxlen=buffer_size)

    # Optimizer
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=learning_rate)
    epsilon = 1.0  # Start with full exploration
    epsilon_min = 0.01
    epsilon_decay = 0.9995  # Decrease exploration over time
    # Loss function
    loss_fn = torch.nn.MSELoss()
    iterations = 0

    while True:
        if iterations > 10000:
            break
        done = False
        reward = None
        round_len = 0
        rand = 0
        network = 0
        l_a = 0
        r_a = 0
        while not done:
            env.render()
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
            # print(result)
            
            # x = math.degrees(result[0][2])
            # print("Pole angle:", x)
            next_state, reward, done, _, _ = result  
            replay_buffer.append((state, action, reward, next_state, done))
            if len(replay_buffer) > batch_size * 10:
                train_model(replay_buffer, policy_net, optimizer, loss_fn)
            # time.sleep(0.5)
            state = next_state
            round_len += 1
        print("RESTART @ ROUND LENGTH:", round_len)
        print("Ratio of random actions:", rand/(rand+network))
        print("Ration of round length to network actions:", round_len/network)
        print("Ratio of network left actions to right actions:", l_a/r_a)
        # time.sleep(1)
        iterations += 1
        print("Iterations:", iterations)
        epsilon *= epsilon_decay
        print("Epsilon:", epsilon)
        if epsilon < epsilon_min:
            epsilon = epsilon_min
        state, _ = env.reset()
        state = np.array(state)


if __name__ == "__main__":
    main()