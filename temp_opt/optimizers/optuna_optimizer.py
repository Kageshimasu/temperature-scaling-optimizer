import torch
import optuna

from ..label_stores.logits_and_labels_store import LogitsAndLabelsStore
from ..trainers.temperature_scale_trainer import TemperatureScaleTrainer


class OptunaOptimizer:

    def __init__(
            self,
            label_store: LogitsAndLabelsStore,
            temperature_scaler: TemperatureScaleTrainer,
            criterion,
            t_range=[1e-1, 5.0],
            max_iter=500):
        """
        :param label_store:
        :param temperature_scaler:
        :param criterion:
        :param max_iter:
        """
        self._label_store = label_store
        self._trainer = temperature_scaler
        self.temperature = self._trainer.get_parameters()
        self._optimizer = optuna.create_study()
        self._criterion = criterion
        self._t_range = t_range
        self._max_iter = max_iter

    def run(self):
        logits, labels = self._label_store.get_logits_and_labels()

        def _target_func(trial):
            if trial.trial_id == 0:
                t = trial.suggest_uniform('t', 1.0, 1.0)
            else:
                t = trial.suggest_uniform('t', self._t_range[0], self._t_range[1])
            self._trainer.set_temperature(float(t))
            loss = self._criterion(self._trainer(logits), labels)
            return loss

        print('START OPTIMIZATION...')
        current_loss = self._evaluate(self._trainer(logits), labels)
        print('before Temperature scaling Loss: {}'.format(current_loss))
        self._optimizer.optimize(_target_func, n_trials=self._max_iter)
        self._trainer.set_temperature(self._optimizer.best_params['t'])
        current_loss = self._evaluate(self._trainer(logits), labels)
        print('after temperature scaling Loss: {}'.format(current_loss))
        print('the best parameter would be {}'.format(self._trainer.get_temperature()))
        return self._trainer.get_temperature()

    def _evaluate(self, logits, labels):
        with torch.no_grad():
            current_loss = float(self._criterion(logits, labels))
        return current_loss
