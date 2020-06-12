import torch
import numpy as np
from matplotlib import pyplot as plt
from torch.nn import functional as F
from temp_opt.label_stores.simple_label_store import LogitsAndLabelsStore


class CalibationPlotter:
    _X = 'confidence'
    _Y = 'accuracy'

    def __init__(self, n_bins=15):
        self._n_bins = n_bins
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def plot(self, label_store: LogitsAndLabelsStore):
        logits, labels = label_store.predict_all()
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)
        fig, ax = plt.subplots()
        ece = 0
        name_list = []
        accuracy_list = []
        calibrated_list = []

        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean().item()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += (torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin).item()
                accuracy_list.append(accuracy_in_bin)
                calibrated_list.append((bin_lower.item() + bin_upper.item()) / 2)
                name_list.append('[{}, {}]'.format(round(bin_lower.item(), 2), round(bin_upper.item(), 2)))
        well_calibrated_x = np.linspace(0, self._n_bins, 100)
        well_calibrated_y = np.linspace(0, 1.0, 100)
        ax.bar(name_list, calibrated_list, color='salmon', edgecolor='r', linewidth=3, alpha=0.7, label='Gap')
        ax.bar(name_list, accuracy_list, color='b', label='Outputs')
        ax.plot(well_calibrated_x, well_calibrated_y, color="grey", linewidth=3.0, linestyle='dashed')
        t = ax.text(0.4, 0.05, 'ERROR=' + str(round(ece, 2)), transform=ax.transAxes, fontsize=30)
        t.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='grey'))
        ax.legend(loc='upper left', borderaxespad=1, fontsize=18)
        ax.set_xlabel(self._X)
        ax.set_ylabel(self._Y)
        ax.set_xlim([0.0, self._n_bins])
        ax.set_ylim([0.0, 1.0])

