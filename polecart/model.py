import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from .params import batch_size, gamma

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)  # First hidden layer
        self.fc2 = nn.Linear(64, 64)         # Second hidden layer
        self.fc3 = nn.Linear(64, action_dim) # Output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))   # Apply ReLU activation
        x = F.relu(self.fc2(x))   # Apply ReLU activation
        x = self.fc3(x)           # Output layer
        return F.softmax(x, dim=-1)  # Output action probabilities


    # When training, pay special attention to terminal states
def train_model(replay_buffer, policy_net, optimizer, loss_fn):
    # Sample a batch of transitions from the replay buffer
    batch = random.sample(replay_buffer, batch_size)
    state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
    print("Done batch 1:", done_batch)
    # Convert to tensors
    state_batch = torch.FloatTensor(state_batch)
    action_batch = torch.LongTensor(action_batch)
    reward_batch = torch.FloatTensor(reward_batch)
    next_state_batch = torch.FloatTensor(next_state_batch)
    done_batch = torch.FloatTensor(done_batch)
    print("done batch:", done_batch)
    # Compute Q values for current states
    current_q_values = policy_net(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
    print("Current q values:", current_q_values)
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
