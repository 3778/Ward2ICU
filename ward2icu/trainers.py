"""
References:
    - https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/gan/gan.py
"""
import torch
import numpy as np
import torchgan
from torch.nn import BCELoss, BCEWithLogitsLoss
from ward2icu import make_logger
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef

logger = make_logger(__file__)


class BinaryClassificationTrainer(object):
    def __init__(self,
                 model,
                 optimizer,
                 sampler_train=None,
                 sampler_test=None,
                 log_to_mlflow=True,
                 loss_function=BCEWithLogitsLoss(),
                 metrics_prepend=''):
        self.optimizer = optimizer
        self.sampler_train = sampler_train
        self.sampler_test = sampler_test
        self.loss_function = loss_function
        self.model = model
        self.log_to_mlflow = log_to_mlflow
        self.tiled = sampler_train.tile
        self.metrics_prepend = metrics_prepend

    def train(self, epochs, evaluate_interval=100):
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            X_train, y_train = self.sampler_train.sample()

            logits_train = self.model(X_train)
            loss = self.loss_function(logits_train, y_train)
            loss.backward()
            self.optimizer.step()

            if epoch % evaluate_interval == 0:
                with torch.no_grad():
                    X_test, y_test = self.sampler_test.sample()
                    logger.debug(f'[Train sizes {X_train.shape} {y_train.shape}]')
                    logger.debug(f'[Test sizes {X_test.shape} {y_test.shape}]')
                    metrics = self.evaluate(X_test, y_test, 
                                            X_train, y_train)
                    msg = f'[epoch {epoch}]'
                    msg += ''.join(f'[{m} {np.round(v,4)}]' 
                                   for m, v in metrics.items()
                                   if m.endswith('balanced_accuracy') or
                                   m.endswith('matheus'))
                    logger.info(msg)
                    if self.log_to_mlflow:
                        mlflow.log_metrics(metrics, step=epoch)

    def evaluate(self, X_test, y_test, X_train, y_train):
        def _calculate(X, y, name):
            logits = self.model(X)
            probs = torch.sigmoid(logits)
            y_pred = probs.round()
            y_true = y
            return self.calculate_metrics(y_true, y_pred, logits, probs, name)

        mp = self.metrics_prepend
        return {**_calculate(X_test, y_test, f'{mp}test'),
                **_calculate(X_train, y_train, f'{mp}train')}

    def calculate_metrics(self, y_true, y_pred, logits, probs, name=''):
        y_true_ = (y_true[:, 0] if self.tiled else y_true).cpu()
        y_pred_ = (y_pred.mode().values if self.tiled else y_pred).cpu()

        mask_0 = (y_true_ == 0)
        mask_1 = (y_true_ == 1)

        hits = (y_true_ == y_pred_).float()
        bas = balanced_accuracy_score(y_true_, y_pred_)
        matthews = matthews_corrcoef(y_true_, y_pred_)

        return {f'{name}_accuracy': hits.mean().item(), 
                f'{name}_balanced_accuracy': bas,
                f'{name}_accuracy_0': hits[mask_0].mean().item(), 
                f'{name}_accuracy_1': hits[mask_1].mean().item(),
                f'{name}_loss': self.loss_function(logits, y_true).item(),
                f'{name}_loss_0': self.loss_function(logits[mask_0],
                                                     y_true[mask_0]).item(),
                f'{name}_loss_1': self.loss_function(logits[mask_1],
                                                     y_true[mask_1]).item(),
                f'{name}_matthews': matthews}
