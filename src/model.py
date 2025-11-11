import numpy as np
from tinygrad.tensor import Tensor

class DQN:
    def __init__(self, state_dim, action_dim, hidden=64):
        self.W1 = Tensor.uniform(state_dim, hidden)
        self.b1 = Tensor.zeros(hidden)
        self.W2 = Tensor.uniform(hidden, hidden)
        self.b2 = Tensor.zeros(hidden)
        self.W3 = Tensor.uniform(hidden, action_dim)
        self.b3 = Tensor.zeros(action_dim)

    def __call__(self, x):
        x = Tensor(x).reshape(1, -1)
        h1 = x.dot(self.W1).add(self.b1).relu()
        h2 = h1.dot(self.W2).add(self.b2).relu()
        out = h2.dot(self.W3).add(self.b3)
        return out

    def parameters(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]
