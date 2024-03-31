import torch
import torch.nn as nn
import torch.optim as optim
import pytest
from pathlib import Path
# append sys path
import sys
sys.path.append(Path(__file__).resolve().parents[1].as_posix())

from linear import apply_bitlinear

# Define a simple linear model
class SimpleLinearModel(nn.Module):
    def __init__(self, bit=False):
        super(SimpleLinearModel, self).__init__()
        self.linear = nn.Linear(1, 1, bias=False)
        if bit:
            apply_bitlinear(self)
            
    def forward(self, x):
        return self.linear(x)
# "bit", [False, True], "trend", [-1, 1], "initializer", [0.1, -0.1, 0.5])
# use matrix for all combinations

import itertools

@pytest.mark.parametrize("bit, trend, initializer", itertools.product([False, True], [-1, 1], [0.1, -0.1, 0.5]))
def test_convergence(bit: bool, trend: int, initializer: float):
    # Set a known seed for reproducibility
    torch.manual_seed(42)

    # Create a dataset
    x = torch.rand(100, 1) * 10  # 100 random values between 0 and 10
    def to_y(x):
        return trend * 0.3 * x
    y = to_y(x) # + 2  # y = 3x + 2

    # Initialize the model
    model = SimpleLinearModel(bit=bit)
    # initialize weight
    model.linear.weight.data.fill_(initializer)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Training loop
    for epoch in range(1000):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

    # Check if the model's parameters are close to the actual values
    # learned_weight = model.linear.weight
    # learned_bias = model.linear.bias.item()
    if bit:
        # check same sign
        assert model.linear.weight.sign() == to_y(torch.tensor([[1.0]])).sign(), "Weight did not converge"
    else:
        assert torch.isclose(model(torch.tensor([[2.0]])), to_y(torch.tensor([[2.0]]))), "Weight did not converge"
    

# Run the test
if __name__ == '__main__':
    test_convergence(bit=False, trend=1)
    test_convergence(bit=True, trend=1)
