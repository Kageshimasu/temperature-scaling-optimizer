import torch
import torchvision
import matplotlib.pyplot as plt
import torchvision.models as models

from temp_opt.label_stores.predicting_table import PredictingTable
from temp_opt.label_stores.simple_label_store import LogitsAndLabelsStore
from temp_opt.predictors.simple_temperature_predictor import TemperatureScalePredictor
from temp_opt.visualizers.calibration_plotter import CalibationPlotter


def main():
    data_path = './data'
    batch_size = 34
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    dataset = torchvision.datasets.CIFAR100(data_path, transform=transforms, train=True, download=True)
    model = models.resnet18(pretrained=True)
    model.load_state_dict(torch.load('trained_model.pth'))
    model_dict = {
        TemperatureScalePredictor(model, 10.2): torch.utils.data.DataLoader(dataset, batch_size=batch_size),
        # models.resnet18(pretrained=True): torch.utils.data.DataLoader(dataset, batch_size=batch_size),
        # models.resnet18(pretrained=True): torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    }
    label_store = LogitsAndLabelsStore(PredictingTable(model_dict))
    plotter = CalibationPlotter()
    plotter.plot(label_store)
    plt.show()


if __name__ == '__main__':
    main()
