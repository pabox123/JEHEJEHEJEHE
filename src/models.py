import torch
import torch.nn as nn
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