import torch
import torch.nn as nn
import torch.nn.functional as F

from mopi.type import PytorchConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Decoder(nn.Module):
    def __init__(self, config: PytorchConfig):
        super(Decoder, self).__init__()

        self.id = id
        self.hidden_size = config.hidden_size

        self.embedding = nn.Embedding(config.output_size, config.hidden_size)
        self.out = nn.Linear(config.hidden_size, config.output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output = self.softmax(self.out(output[0]))
        return output

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
