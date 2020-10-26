# Temperature Scaling Optimizer

Temperature Scaling Optimizer is to calibrate your neural network and 
visualize how well-calibrated it in Pytorch.  
This library is based on the below papers.
- [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599).
- [temperature_scaling](https://github.com/gpleiss/temperature_scaling)

## Table of Contents

1. [Features](#features)
2. [Installation](#installation)
3. [Examples](#examples)

<a name="features"/>

## Features
* Optimize for calibrating your neural network compatible with Cross Validation
* Visualize how well-calibrated your neural network is

<a name="installation"/>

## Installation
```
pip install -r requirements.txt
```
And you also install pytorch from [here](https://pytorch.org/)

<a name="examples"/>

## Examples
To Optimize
```python
import temp_opt as topt

model_dict = {
        model_1: DataLoader_1,
        model_2: DataLoader_2,
        model_3: DataLoader_3
        }
label_store = topt.label_stores.LogitsAndLabelsStore(topt.label_stores.PredictingTable(model_dict))
lbfgs_opt = topt.optimizers.LBFGSOptimizer(label_store, topt.trainers.TemperatureScaleTrainer())
lbfgs_opt.run()
```

To Predict with Temperature Scaling
```python
import torch
import torchvision.models as models
import temp_opt as topt

model = models.resnet18(pretrained=True)
temperature = 5.32  # set an optimized temperature value 
predictor = topt.predictors.TemperatureScalePredictor(model, temperature)
inputs = torch.Tensor(34, 3, 32, 32)
print(predictor(inputs))
```

To Visualize
```python
import matplotlib.pyplot as plt
import temp_opt as topt

model_dict = {
        model_1: DataLoader_1,
        model_2: DataLoader_2,
        model_3: DataLoader_3
        }
label_store = topt.label_stores.LogitsAndLabelsStore(topt.label_stores.PredictingTable(model_dict))
plotter = topt.visualizers.CalibationPlotter()
plotter.plot(label_store)
plt.show()
```
You can visualize your neural network as in the diagram below
![Visualize Sample](https://github.com/Kageshimasu/temperature-scaling-optimizer/blob/master/images/calibrated_result.png)
