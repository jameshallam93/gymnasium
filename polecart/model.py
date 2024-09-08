import random
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from params import batch_size, gamma


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)  # First hidden layer
        self.fc2 = nn.Linear(64, 64)  # Second hidden layer
        self.fc3 = nn.Linear(64, action_dim)  # Output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Apply ReLU activation
        x = F.relu(self.fc2(x))  # Apply ReLU activation
        x = self.fc3(x)  # Output layer
        return F.softmax(x, dim=-1)  # Output action probabilities

    # When training, pay special attention to terminal states


def train_model(replay_buffer, policy_net, optimizer, loss_fn):
    # Sample a batch of transitions from the replay buffer
    batch = random.sample(replay_buffer, batch_size)
    state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
    # Convert to tensors
    state_batch = torch.FloatTensor(np.array(state_batch))
    action_batch = torch.LongTensor(action_batch)
    reward_batch = torch.FloatTensor(reward_batch)
    next_state_batch = torch.FloatTensor(next_state_batch)
    done_batch = torch.FloatTensor(done_batch)
    # Compute Q values for current states
    current_q_values = (
        policy_net(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
    )
    # Compute target Q values using Bellman equation
    with torch.no_grad():
        # If done, the future Q-value is 0 (no more rewards after termination)
        max_next_q_values = policy_net(next_state_batch).max(1)[0]
        target_q_values = reward_batch + gamma * max_next_q_values * (1 - done_batch)

    # Compute loss (MSE between predicted Q values and target Q values)
    loss = loss_fn(current_q_values, target_q_values)

    # Backpropagate the loss and update the network
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def save_model(policy_net, filename):
    torch.save(policy_net.state_dict(), filename)
    print(f"Model saved to {filename}")


def get_action(env, policy_net, state, reporting):
    # use a random action with probability epsilon for exploration
    if random.random() < reporting.epsilon.value:
        action = env.action_space.sample()  # Random action for exploration
        reporting.rand += 1
    else:
        action = policy_net(torch.FloatTensor(state).unsqueeze(0)).argmax().item()
        reporting.network += 1
        if action == 0:
            reporting.l_a += 1
        else:
            reporting.r_a += 1
    return action


def run_non_training(env, policy_net, state, debug=False):
    # Preprocess the state as needed
    state_tensor = torch.FloatTensor(state).unsqueeze(
        0
    )  # Convert to tensor and add batch dimension

    # Use the loaded model to select an action
    action = policy_net(state_tensor).argmax().item()
    if debug:
        print("Action:", "left" if action == 0 else "right")

    # Take the selected action in the environment
    next_state, reward, done, _, _ = env.step(action)

    # Update the current state
    env.render()  # render the environment to visualize the agent's behavior
    return next_state, reward, done
