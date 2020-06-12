# Tempreture Scaling Optimizer

Tempreture Scaling Optimizer is to calibrate your neural network and 
visualize how well-calibrated it.  
This library is based on below pages
- [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599).
- [temperature_scaling](https://github.com/gpleiss/temperature_scaling)

## Table of Contents

1. [Features](#features)
2. [Installation](#installation)
3. [Examples](#examples)

<a name="features"/>

## Features
* Optimize for calibrating your neural network compatible with Cross Validation
* Visualize how well-calibrated your neural network

<a name="installation"/>

## Installation
UNDER CONSTRUCTION...

<a name="examples"/>

## Examples
To Optimize
```python
from temp_opt.label_stores.predicting_table import PredictingTable
from temp_opt.label_stores.simple_label_store import LogitsAndLabelsStore
from temp_opt.trainers.temperature_scale_trainer import TemperatureScaleTrainer
from temp_opt.optimizers.lbfgs_optimizer import LBFGSOptimizer

model_dict = {
        model_1: DataLoader_1,
        model_2: DataLoader_2,
        model_3: DataLoader_3
        }
label_store = LogitsAndLabelsStore(PredictingTable(model_dict))
lbfgs_opt = LBFGSOptimizer(label_store, TemperatureScaleTrainer())
lbfgs_opt.run()
```

To Visualize
```python
from temp_opt.label_stores.predicting_table import PredictingTable
from temp_opt.label_stores.simple_label_store import LogitsAndLabelsStore
from temp_opt.visualizers.calibration_plotter import CalibationPlotter

model_dict = {
        model_1: DataLoader_1,
        model_2: DataLoader_2,
        model_3: DataLoader_3
        }
label_store = LogitsAndLabelsStore(PredictingTable(model_dict))
plotter = CalibationPlotter()
plotter.plot(label_store)
plt.show()
```

To Predict with Temperature Scaling
```python
from temp_opt.predictors.simple_temperature_predictor import TemperatureScalePredictor

model = models.resnet18(pretrained=True)
tempereture = 5.32  # set an optimized tempreture value 
predictor = TemperatureScalePredictor(model, tempereture)
inputs = torch.Tensor(34, 3, 32, 32)
print(predictor(inputs))
```
