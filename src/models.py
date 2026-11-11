import torch
import torch.nn as nn
from typing import List

class DenseNN(nn.Module):
    #dense Neural Network for binary sentiment classification

    def __init__(
            self,
            input_dim: int,
            hidden_dims: List[int],
            output_dim: int = 1,
            dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()

        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one dimension.")

        layers: List[nn.Module] = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


if __name__ == "__main__":
    batch_size = 16
    input_dim = 128
    hidden_dims = [64, 32]

    model = DenseNN(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=1)
    dummy_input = torch.randn(batch_size, input_dim)
    output = model(dummy_input)

    print(f"Input shape : {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model:\n{model}")