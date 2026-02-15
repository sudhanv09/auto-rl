from typing import Tuple
import time
from tinygrad import Tensor, TinyJit, nn
from tinygrad.helpers import trange
import numpy as np
from collections import deque
import random

from model import DQN
from env import Env

OBSERVATION_DIM = 7
ACTION_DIM = 10
MIN_PAYOUT = 15.0
MAX_PAYOUT = 60.0

LEARNING_RATE = 1e-3
BATCH_SIZE = 32
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
REPLAY_BUFFER_SIZE = 10000
TARGET_UPDATE_FREQ = 100
TRAIN_STEPS = 5

class ReplayBuffer:
    """Simple experience replay buffer for DQN."""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self):
        return len(self.buffer)

def action_to_payout(action: int) -> float:
    payout = MIN_PAYOUT + (action / (ACTION_DIM - 1)) * (MAX_PAYOUT - MIN_PAYOUT)
    return payout

def select_action(model: DQN, state: np.ndarray, epsilon: float) -> int:
    if random.random() < epsilon:
        return random.randint(0, ACTION_DIM - 1)
    
    state_tensor = Tensor(state.astype(np.float32))
    q_values = model(state_tensor).numpy()
    
    return int(np.argmax(q_values))

if __name__ == "__main__":
    env = Env()
    model = DQN(OBSERVATION_DIM, ACTION_DIM)
    target_model = DQN(OBSERVATION_DIM, ACTION_DIM)
    
    # Copy initial weights
    for param, target_param in zip(nn.state.get_parameters(model), nn.state.get_parameters(target_model)):
        target_param.assign(param)
    
    opt = nn.optim.Adam(nn.state.get_parameters(model), lr=LEARNING_RATE)
    replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
    
    @TinyJit
    def train_step(x: Tensor, selected_action: Tensor, reward: Tensor, next_x: Tensor, done: Tensor) -> Tensor:
        with Tensor.train():
            # Current Q-values
            q_values = model(x)
            action_mask = (selected_action.reshape(-1, 1) == Tensor.arange(q_values.shape[1]).reshape(1, -1).expand(selected_action.shape[0], -1)).float()
            q_value = (q_values * action_mask).sum(1)
            
            # Next Q-values from target network (detach to prevent gradients)
            next_q_values = target_model(next_x).detach()
            next_q_value = next_q_values.max(1)
            target_q_value = reward + (1 - done) * GAMMA * next_q_value
            
            # Compute loss (MSE)
            loss = (q_value - target_q_value).square().mean()
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            return loss.realize()
    
    @TinyJit
    def get_action(obs: Tensor) -> Tensor:
        ret = model(obs).argmax().realize()
        return ret
    
    st, steps = time.perf_counter(), 0
    epsilon = EPSILON_START
    total_steps = 0
    
    for episode_number in (t := trange(500)):
        get_action.reset()
        
        state = env.reset()
        rews, done = [], False
        
        while not env.is_done():
            if random.random() < epsilon:
                act = random.randint(0, ACTION_DIM - 1)
            else:
                act = get_action(Tensor(state)).item()
            
            payout = action_to_payout(act)
            next_state, rew, done, info = env.step(payout)
            
            replay_buffer.push(state, act, rew, next_state, done)
            
            state = next_state
            rews.append(float(rew))
            steps += 1
            total_steps += 1
        
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        
        if len(replay_buffer) >= BATCH_SIZE:
            states_list, actions_list, rewards_list, next_states_list, dones_list = zip(*replay_buffer.buffer)
            X = Tensor(np.array(states_list).astype(np.float32))
            A = Tensor(np.array(actions_list).astype(np.int32))
            R = Tensor(np.array(rewards_list).astype(np.float32))
            Xn = Tensor(np.array(next_states_list).astype(np.float32))
            D = Tensor(np.array(dones_list).astype(np.float32))
            
            for i in range(TRAIN_STEPS):
                if len(replay_buffer) > BATCH_SIZE:
                    samples = Tensor.randint(BATCH_SIZE, high=X.shape[0]).realize()
                    loss = train_step(X[samples], A[samples], R[samples], Xn[samples], D[samples])
            
            if total_steps % TARGET_UPDATE_FREQ == 0:
                for param, target_param in zip(nn.state.get_parameters(model), nn.state.get_parameters(target_model)):
                    target_param.assign(param)
        
        t.set_description(
            f"sz: {len(replay_buffer):5d} steps/s: {steps/(time.perf_counter()-st):7.2f} "
            f"eps: {epsilon:.3f} reward: {sum(rews):6.2f} funds: {info.get('funds', 0):.0f}"
        )
