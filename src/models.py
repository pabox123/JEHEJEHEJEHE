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
from transformers import DistilBertModel


class TransformerSentiment(nn.Module):
    """

    Args:
        model_name (str): Name of the pre-trained transformer model
        num_classes (int): Number of output classes
        dropout (float): Dropout rate for the classification head
    """

    def __init__(self, model_name='distilbert-base-uncased', num_classes=2, dropout=0.3):
        super(TransformerSentiment, self).__init__()

        self.distilbert = DistilBertModel.from_pretrained(model_name)

        hidden_size = self.distilbert.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls_output = outputs.last_hidden_state[:, 0, :]

        logits = self.classifier(cls_output)

        return logits
