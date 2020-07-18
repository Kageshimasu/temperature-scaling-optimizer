import torch
import tqdm
from typing import Tuple

from ..label_stores.predicting_table import PredictingTable


class LogitsAndLabelsStore:

    def __init__(self, predicting_table: PredictingTable):
        """
        :param predicting_table:
        """
        self._predicting_table = predicting_table
        self._logits, self._labels = self._predict_all()

    def get_logits_and_labels(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._logits, self._labels

    def _predict_all(self) -> Tuple[torch.Tensor, torch.Tensor]:
        logits_list = []
        labels_list = []
        i = 0
        t = 0
        with torch.no_grad():
            for model, val_loader in self._predicting_table:
                for input, label in tqdm.tqdm(val_loader):
                    if torch.cuda.is_available():
                        input = input.cuda()
                        label = label.cuda()
                        model.cuda()
                    logit = model(input)
                    if torch.sum(logit[0]).item() == 1.0:
                        print('WARNING: Check if you use softmax in your model')
                    logits_list.append(logit)
                    if len(label.shape) == 2:
                        labels_list.append(torch.argmax(label, dim=1))
                    else:
                        labels_list.append(label)
                    t += 1
                    # if t > 1000:
                    #     break
                i += 1
            logits = torch.cat(logits_list).cuda()
            labels = torch.cat(labels_list).cuda()
        return logits, labels
