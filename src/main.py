import numpy as np
from collections import deque
import tinygrad.nn as nn
from tinygrad.tensor import Tensor
from tinygrad import Device

from model import DQN
from env import Env

EPISODES = 1000
EPSILON = 0.2
GAMMA = 0.95
LR = 0.001
BATCH_SIZE = 32
TARGET_UPDATE = 10
REPLAY_MEMORY = 2000

env = Env(
    {
        "driver_count": 50,
        "seed_money": 1_000_000,
        "base_demand": 1000,
        "multiplier": 50,
        "avg_order_fee_per_km": 30,
        "initial_driver_count": 50,
    }
)

state = env.reset()
state_dim = len(state)

rate_options = np.arange(25, 61, 5)
action_dim = len(rate_options)

policy_net = DQN(state_dim, action_dim)
target_net = DQN(state_dim, action_dim)
for p, t in zip(policy_net.parameters(), target_net.parameters()):
    t.assign(p)

opt = nn.optim.Adam(policy_net.parameters(), lr=LR)
memory = deque(maxlen=REPLAY_MEMORY)

print(f"Running on: {Device.DEFAULT}")

for episode in range(EPISODES):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        # ε-greedy policy
        if np.random.random() < EPSILON:
            action = np.random.randint(0, action_dim - 1)
        else:
            Tensor.training = False
            q_values = policy_net(state)
            Tensor.training = True 
            action = int(q_values.argmax().item())

        # choose how much to pay
        rate = float(rate_options[action])
        policy = {"rate_per_km": rate}

        next_state, reward, done, _ = env.step(policy)
        memory.append((state, action, reward, next_state, done))

        if len(memory) > BATCH_SIZE:
            indices = np.random.choice(len(memory), BATCH_SIZE, replace=False)
            batch = [memory[i] for i in indices]

            states, actions, rewards, next_states, dones = zip(*batch)

            losses = []
            for i in range(BATCH_SIZE):
                Tensor.training = False
                target_q = rewards[i] + (1 - dones[i]) * GAMMA * target_net(next_states[i]).max().item()
                Tensor.training = True

                q_pred = policy_net(states[i])[0, actions[i]]
                losses.append((q_pred - target_q) ** 2)

            loss = Tensor.stack(losses).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

        total_reward += reward
        state = next_state

    if episode % TARGET_UPDATE == 0:
        for p, t in zip(policy_net.parameters(), target_net.parameters()):
            t.assign(p)

    num_drivers_quit = int(env.company.driver_churn_rate * len(env.drivers))
    incomplete_orders = env.state.total_orders_available - env.state.total_orders_complete

    print(f"Episode {episode}:")
    print(f"  Total Reward        = {total_reward:.2f}")
    print(f"  Company Profit      = {env.company.last_epoch_profit:.2f}")
    print(f"  Avg Driver Profit   = {env.company.avg_driver_profit:.2f}")
    print(f"  Drivers Quit        = {num_drivers_quit}")
    print(f"  Orders Incomplete   = {incomplete_orders}")
    print("-" * 50)

