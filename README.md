<p align="center">
  <h1 align="center">Temperature Scaling Optimizer</h1>
</p>
<p align="center">
    <em>A PyTorch library to calibrate your model's confidence and make it more reliable.</em>
</p>


---

Modern neural networks are often poorly calibrated, meaning their confidence scores don't accurately reflect the true probability of being correct. **Temperature Scaling Optimizer** solves this by finding the optimal temperature to scale your model's logits, making its predictions more reliable without sacrificing accuracy.

This library is based on the foundational work presented in:
- [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599)
- [gpleiss/temperature_scaling](https://github.com/gpleiss/temperature_scaling)

## ✨ Features

* **Find Optimal Temperature**: Automatically finds the best temperature for calibration, with built-in support for cross-validation.
* **Visualize Calibration**: Generates reliability diagrams to let you visually assess how well-calibrated your model is.
* **PyTorch Native**: Seamlessly integrates with your existing PyTorch models and `DataLoader`s.

## 🚀 Installation

### Prerequisites
- Python 3.8+
- PyTorch

First, install PyTorch by following the official instructions on their [website](https://pytorch.org/).

### Install from Source
```bash
git clone [https://github.com/your-username/temperature-scaling-optimizer.git](https://github.com/your-username/temperature-scaling-optimizer.git)
cd temperature-scaling-optimizer
pip install -e .
```

## 💡 Quick Start

Here’s a complete workflow from optimization to visualization.

### 1. Setup
First, prepare your trained models and validation dataloaders. For cross-validation, you would typically have a model and a dataloader for each fold.

```python
import torch
import torchvision.models as models
from torch.utils.data import DataLoader, TensorDataset
import temp_opt as topt

# Example: Create placeholder models and dataloaders for 3 CV folds
# In a real scenario, these would be your actual trained models and validation sets.
model_1 = models.resnet18(pretrained=False) # Assume fold 1 model
model_2 = models.resnet18(pretrained=False) # Assume fold 2 model
model_3 = models.resnet18(pretrained=False) # Assume fold 3 model

# Create dummy dataloaders
dummy_data = torch.randn(100, 3, 32, 32)
dummy_labels = torch.randint(0, 10, (100,))
dataloader_1 = DataLoader(TensorDataset(dummy_data, dummy_labels), batch_size=32)
dataloader_2 = DataLoader(TensorDataset(dummy_data, dummy_labels), batch_size=32)
dataloader_3 = DataLoader(TensorDataset(dummy_data, dummy_labels), batch_size=32)

# The model_dict maps each model to its corresponding validation data.
model_dict = {
    model_1: dataloader_1,
    model_2: dataloader_2,
    model_3: dataloader_3,
}
```

### 2. Find the Optimal Temperature
Use the optimizer to find the best temperature value based on your validation data.

```python
# Store the logits and labels from all validation folds
label_store = topt.label_stores.LogitsAndLabelsStore(
    topt.label_stores.PredictingTable(model_dict)
)

# Initialize and run the optimizer to find the single best temperature
lbfgs_opt = topt.optimizers.LBFGSOptimizer(label_store, topt.trainers.TemperatureScaleTrainer())
optimal_temperature = lbfgs_opt.run()

print(f"✅ Optimal Temperature found: {optimal_temperature:.4f}")
```

### 3. Apply Temperature Scaling for Prediction
Use the `optimal_temperature` to make calibrated predictions with any of your models.

```python
# Assuming 'model' is the model you want to use for inference
model_for_inference = models.resnet18(pretrained=True)
model_for_inference.eval()

# Create a predictor with the optimized temperature
predictor = topt.predictors.TemperatureScalePredictor(model_for_inference, temperature=optimal_temperature)

# Get calibrated probabilities
inputs = torch.randn(1, 3, 224, 224)
calibrated_probs = predictor(inputs)
print("Calibrated probabilities:\n", calibrated_probs)
```

### 4. Visualize Calibration
Plot a reliability diagram to see the impact of calibration.

```python
import matplotlib.pyplot as plt

plotter = topt.visualizers.CalibationPlotter()

# 1. Plot the reliability diagram for the uncalibrated model
print("Plotting uncalibrated results...")
plotter.plot(label_store, title="Before Temperature Scaling")
plt.show()

# 2. Apply temperature scaling to the stored logits
label_store.scale_logits(optimal_temperature)

# 3. Plot the now-calibrated results
print("Plotting calibrated results...")
plotter.plot(label_store, title=f"After Temperature Scaling (T={optimal_temperature:.3f})")
plt.show()
```

This generates a clear visual comparison, showing how T-scaling improves the alignment between model confidence and accuracy.

![Visualize Sample](https://github.com/Kageshimasu/temperature-scaling-optimizer/blob/master/images/calibrated_result.png)

---

## 🤝 Contributing

Contributions are welcome! Whether it's bug reports, feature requests, or pull requests, please feel free to engage with the project. Please open an issue to discuss any significant changes beforehand.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
