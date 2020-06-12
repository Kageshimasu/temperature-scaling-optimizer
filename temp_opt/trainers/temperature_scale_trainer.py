import torch
from torch import nn


class TemperatureScaleTrainer(nn.Module):

    def __init__(self):
        super(TemperatureScaleTrainer, self).__init__()
        self._temperature = nn.Parameter(torch.ones(1), requires_grad=True)
        if torch.cuda.is_available():
            self._temperature.cuda()
            self.cuda()

    def forward(self, logits: torch.Tensor):
        expanded_temperature = self._temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / expanded_temperature

    def get_parameters(self):
        return self._temperature

    def get_temperature(self):
        return self._temperature.item()
