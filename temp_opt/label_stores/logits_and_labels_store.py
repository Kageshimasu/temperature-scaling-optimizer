import torch
import tqdm
import cv2

from ..label_stores.predicting_table import PredictingTable


class LogitsAndLabelsStore:

    def __init__(self, predicting_table: PredictingTable):
        """
        :param predicting_table:
        """
        self._predicting_table = predicting_table

    def predict_all(self):
        """
        :return:
        """
        logits_list = []
        labels_list = []
        i = 0
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
                    labels_list.append(label)
                i += 1
            logits = torch.cat(logits_list).cuda()
            labels = torch.cat(labels_list).cuda()
        return logits, labels
