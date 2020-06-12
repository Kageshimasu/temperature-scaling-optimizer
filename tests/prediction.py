import torch
import torchvision.models as models
from temp_opt.predictors.simple_temperature_predictor import TemperatureScalePredictor


def main():
    model = models.resnet18(pretrained=True)
    predictor = TemperatureScalePredictor(model, 5.34)
    inputs = torch.Tensor(34, 3, 32, 32)
    print(predictor(inputs))


if __name__ == '__main__':
    main()
