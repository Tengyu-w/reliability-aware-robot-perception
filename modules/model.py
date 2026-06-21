# model.py
import torch
import torch.nn as nn
from torchvision import models

class SimpleCNNLSTM(nn.Module):
    """Simplified CNN+LSTM model"""

    def __init__(self, num_classes, hidden_dim=256, lstm_layers=1):
        super(SimpleCNNLSTM, self).__init__()

        # Use pretrained ResNet. Keep compatibility with older torchvision.
        try:
            weights = models.ResNet18_Weights.DEFAULT
            resnet = models.resnet18(weights=weights)
        except AttributeError:
            resnet = models.resnet18(pretrained=True)

        # Remove the final fully connected layer
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])

        # LSTM
        self.lstm = nn.LSTM(
            input_size=512,  # ResNet18 output dimension
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=False
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def extract_embeddings(self, x):
        # x: [batch, seq, 3, H, W]
        batch_size, seq_len, c, h, w = x.shape

        # CNN feature extraction
        cnn_input = x.view(batch_size * seq_len, c, h, w)
        cnn_features = self.cnn(cnn_input)
        cnn_features = cnn_features.view(batch_size, seq_len, -1)

        # LSTM
        lstm_out, _ = self.lstm(cnn_features)

        # Use the last time step
        last_output = lstm_out[:, -1, :]
        return last_output

    def forward(self, x, return_embedding=False):
        embedding = self.extract_embeddings(x)

        # Classification
        output = self.classifier(embedding)

        if return_embedding:
            return output, embedding
        return output
