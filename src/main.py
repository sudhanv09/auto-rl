import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

from model import DQN
from env import Env

EPISODES = 1000
EPSILON = 0.2
GAMMA = 0.95
LR = 0.001
BATCH_SIZE = 32
TARGET_UPDATE = 10
REPLAY_MEMORY = 2000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize(s: np.ndarray) -> np.ndarray:
    if s.ndim == 1:
        s = np.expand_dims(s, axis=0)

    return np.array(
        [
            s[:, 0] / 1e6,  # balance scaled by 1M
            s[:, 1] / 1e5,  # last_epoch_profit
            s[:, 2] / 10,  # consecutive_loss_epochs
            s[:, 3] / 500,  # active drivers
            s[:, 4],  # churn rate already in [0,1]
            s[:, 5] / 100,  # new_signups
            s[:, 6] / 1e5,  # avg_driver_earnings
            s[:, 7] / 1e5,  # avg_driver_profit
            s[:, 8],  # completion rate in [0,1]
        ],
        dtype=np.float32,
    ).T


env = Env(
    {
        "driver_count": 50,
        "seed_money": 1_000_000,
        "base_demand": 1000,
        "multiplier": 50,
        "order_fee": np.random.randint(10, 60),
        "initial_driver_count": 50,
    }
)

state = env.reset()
state_dim = len(state)

payout_options = np.arange(15, 81, 5)
action_dim = len(payout_options)

policy_net = DQN(state_dim, action_dim).to(DEVICE)
target_net = DQN(state_dim, action_dim).to(DEVICE)
target_net.eval()

opt = optim.Adam(policy_net.parameters(), lr=LR)
memory = deque(maxlen=REPLAY_MEMORY)

print(f"Running on: {DEVICE}")

for episode in range(EPISODES):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        # ε-greedy policy
        if np.random.random() < EPSILON:
            action = np.random.choice(action_dim)
        else:
            with torch.no_grad():
                s = torch.tensor(normalize(state), dtype=torch.float32).to(DEVICE)
                q_values = policy_net(s)
                action = int(torch.argmax(q_values).item())

        # choose how much to pay
        payout = float(payout_options[action])
        policy = {"order_payout": payout}

        next_state, reward, done, _ = env.step(policy)
        memory.append((state, action, reward, next_state, done))

        if len(memory) > BATCH_SIZE:
            indices = np.random.choice(len(memory), BATCH_SIZE, replace=False)
            batch = [memory[i] for i in indices]

            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.tensor(normalize(np.array(states)), dtype=torch.float32).to(DEVICE)
            next_states = torch.tensor(normalize(np.array(next_states)), dtype=torch.float32).to(DEVICE)
            actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(DEVICE)
            rewards = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)
            dones = torch.tensor(dones, dtype=torch.float32).to(DEVICE)

            q_pred = policy_net(states).gather(1, actions).squeeze(1)

            with torch.no_grad():
                q_next = target_net(next_states).max(1)[0]
                q_target = rewards + (1 - dones) * GAMMA * q_next

            loss = nn.functional.mse_loss(q_pred, q_target)

            opt.zero_grad()
            loss.backward()
            opt.step()

        total_reward += reward
        state = next_state

    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    num_drivers_quit = int(env.company.driver_churn_rate * len(env.drivers))
    incomplete_orders = (
        env.state.total_orders_available - env.state.total_orders_complete
    )

    print(f"Episode {episode}:")
    print(f"  Total Reward        = {total_reward:.2f}")
    print(f"  Company Profit      = {env.company.last_epoch_profit:.2f}")
    print(f"  Avg Driver Profit   = {env.company.avg_driver_profit:.2f}")
    print(f"  Drivers Quit        = {num_drivers_quit}")
    print(f"  Orders Incomplete   = {incomplete_orders}")
    print("-" * 50)
