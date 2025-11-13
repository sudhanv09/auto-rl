import numpy as np
from collections import deque
import csv
import os
import datetime

import torch
import torch.nn as nn
import torch.optim as optim

from model import DQN
from env import Env


timestamp = datetime.datetime.now()
LOGFILE = f"logs/progess_{timestamp}.csv"
EPISODES = 1000
EPSILON = 0.2
GAMMA = 0.95
LR = 0.001
BATCH_SIZE = 32
TARGET_UPDATE = 20
REPLAY_MEMORY = 5000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize(s: np.ndarray) -> np.ndarray:
    if s.ndim == 1:
        s = np.expand_dims(s, axis=0)

    return np.array(
        [
            s[:, 0] / 2.0,  # Demand
            s[:, 1] / 50.0,  # backlog
            s[:, 2] / 10.0,  # Profit
            s[:, 3] / 1.0,  # Driver Progress
        ],
        dtype=np.float32,
    ).T


env = Env({"driver_count": 50})

state = env.reset()
state_dim = len(state)
action_dim = len(env.action_space)

policy_net = DQN(state_dim, action_dim).to(DEVICE)
target_net = DQN(state_dim, action_dim).to(DEVICE)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

opt = optim.Adam(policy_net.parameters(), lr=LR)
memory = deque(maxlen=REPLAY_MEMORY)

with open(LOGFILE, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(
        [
            "Episode",
            "Month",
            "Total_Days",
            "Monthly_Profit_TWD",
            "Monthly_Reward",
            "Active_Drivers",
            "Driver_churn",
            "Backlog_Size",
            "Driver_Goal_Progress_Pct",
            "Last_Payout_Ratio",
            "Epsilon",
        ]
    )

print(f"Starting training. Logging to: {os.path.abspath(LOGFILE)}")

print(f"Running on: {DEVICE}")

for episode in range(EPISODES):
    state = env.reset()
    done = False

    # Episode Aggregates
    total_reward = 0
    total_profit_lifetime = 0
    days_survived = 0

    # Monthly Aggregates (Reset every 30 days)
    month_profit = 0
    month_reward = 0
    month_expired_orders = 0

    while not done:
        # ε-greedy policy
        if np.random.random() < EPSILON:
            action = np.random.choice(action_dim)
        else:
            with torch.no_grad():
                s = torch.tensor(normalize(state), dtype=torch.float32).to(DEVICE)
                q_values = policy_net(s)
                action = int(torch.argmax(q_values).item())

        next_state, reward, done, _ = env.step(action)
        memory.append((state, action, reward, next_state, done))

        if len(memory) > BATCH_SIZE:
            indices = np.random.choice(len(memory), BATCH_SIZE, replace=False)
            batch = [memory[i] for i in indices]

            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.tensor(normalize(np.array(states)), dtype=torch.float32).to(
                DEVICE
            )
            next_states = torch.tensor(
                normalize(np.array(next_states)), dtype=torch.float32
            ).to(DEVICE)
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
        total_profit_lifetime += env.company.daily_profit
        days_survived += 1

        month_profit += env.company.daily_profit
        month_reward += reward
        month_expired_orders += env.company.daily_expired

        if env.day_counter % 30 == 0:
            current_month = env.day_counter // 30

            # Get current driver stats
            active_drivers = len([d for d in env.drivers if d.is_active])
            total_drivers = len(env.drivers)
            driver_churn = total_drivers - active_drivers

            # Get observation stats (Driver Progress %)
            # obs[3] is the avg progress toward monthly goal
            driver_happiness_metric = next_state[3] * 100
            backlog = len(env.state.orders_queue)
            current_payout_ratio = env.action_space[action]

            with open(LOGFILE, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        episode,
                        current_month,
                        days_survived,
                        round(month_profit, 2),
                        round(month_reward, 2),
                        active_drivers,
                        driver_churn,
                        backlog,
                        round(driver_happiness_metric, 2),
                        round(current_payout_ratio, 2),
                        round(EPSILON, 4),
                    ]
                )

            # Reset Monthly Trackers
            month_profit = 0
            month_reward = 0
            month_expired_orders = 0

        state = next_state

    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    # Simple epsilon decay (Optional)
    if EPSILON > 0.05:
        EPSILON *= 0.995

    print(
        f"Episode {episode} finished. Survived {days_survived} days"
    )
