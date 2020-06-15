import torch
from torch import nn, optim

from ..label_stores.logits_and_labels_store import LogitsAndLabelsStore
from ..trainers.temperature_scale_trainer import TemperatureScaleTrainer
from ..metrics.eceloss import ECELoss


class LBFGSOptimizer:

    def __init__(
            self,
            label_store: LogitsAndLabelsStore,
            temperature_scaler: TemperatureScaleTrainer,
            criterion=nn.CrossEntropyLoss,
            lr=0.01,
            max_iter=100):
        """
        :param label_store:
        :param temperature_scaler:
        :param criterion:
        :param lr:
        :param max_iter:
        """
        self._label_store = label_store
        self._trainer = temperature_scaler
        self.temperature = self._trainer.get_parameters()
        self._optimizer = optim.LBFGS([self._trainer.get_parameters()], lr=lr, max_iter=max_iter)
        self._criterion = criterion()
        if torch.cuda.is_available():
            self._criterion.cuda()

    def run(self):
        logits, labels = self._label_store.predict_all()

        def _target_func():
            loss = self._criterion(self._trainer(logits), labels)
            loss.backward()
            return loss

        print('START OPTIMIZATION...')
        current_loss, current_cecloss = self._evaluate(self._trainer(logits), labels)
        print('before Temperature scaling Loss: {}, ECELoss: {}'.format(current_loss, current_cecloss))
        self._optimizer.step(_target_func)
        current_loss, current_cecloss = self._evaluate(self._trainer(logits), labels)
        print('after temperature scaling Loss: {}, ECELoss: {}'.format(current_loss, current_cecloss))
        print('the best parameter would be {}'.format(self._trainer.get_temperature()))
        return self._trainer.get_temperature()

    def _evaluate(self, logits, labels):
        ece_criterion = ECELoss()
        with torch.no_grad():
            current_loss = float(self._criterion(self._trainer(logits), labels))
            current_cecloss = float(ece_criterion(self._trainer(logits), labels))
        return current_loss, current_cecloss
