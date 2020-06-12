import torch
from torch import nn


class TemperatureScalePredictor(nn.Module):

    def __init__(self, model: nn.Module, temperature: float):
        super(TemperatureScalePredictor, self).__init__()
        self._model = model
        self._model.eval()
        self._temperature = torch.tensor([temperature])
        if torch.cuda.is_available():
            self._temperature = self._temperature.cuda()
            self._model.cuda()
            self.cuda()

    def forward(self, inputs: torch.Tensor):
        if torch.cuda.is_available():
            inputs = inputs.cuda()
        with torch.no_grad():
            logits = self._model(inputs)
            expanded_temperature = self._temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / expanded_temperature

    def get_temperature(self):
        return self._temperature.item()
