from tinygrad import Tensor
import tinygrad.nn as nn

class DQN:
    def __init__(self, input_dim, output_dim):
        self.linear1 = nn.Linear(input_dim, 64)
        self.linear2 = nn.Linear(64, 128)
        self.linear3 = nn.Linear(128, output_dim)

    def __call__(self, x: Tensor) -> Tensor:
        x = self.linear1(x).relu()
        x = self.linear2(x).relu()
        x = self.linear3(x)
        return x