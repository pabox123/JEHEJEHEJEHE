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

class VanillaRNN(nn.Module):
    #vanilla recurrent neural network binary classification

    def __init__(
            self,
            vocab_size: int,
            embedding_dim: int,
            hidden_dim: int,
            output_dim: int = 1,
            num_layers: int = 1,
            bidirectional: bool = False,
            dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )

        rnn_out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(rnn_out_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.dropout(self.embedding(x))
        output, hidden = self.rnn(embedded)

        if self.rnn.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]

        return self.fc(self.dropout(hidden))


if __name__ == "__main__":
    batch_size = 16
    seq_len = 30
    vocab_size = 10000
    embedding_dim = 64
    hidden_dim = 128

    model = VanillaRNN(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        output_dim=1,
        num_layers=1,
        bidirectional=True,
    )

    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))
    output = model(dummy_input)

    print(f"Input shape : {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model:\n{model}")