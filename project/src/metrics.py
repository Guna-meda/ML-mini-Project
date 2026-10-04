"""Metric collection matching the reference notebook's MyMetric class."""

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from .preprocessing import PRICE_BIN_LABELS


class MyMetric:
    def __init__(self, avg_method: str = "weighted"):
        self.avg_method = avg_method
        self.avg_train_precs = []
        self.avg_train_recalls = []
        self.avg_train_f1s = []
        self.avg_test_precs = []
        self.avg_test_recalls = []
        self.avg_test_f1s = []
        self.per_class_train_precs = []
        self.per_class_test_precs = []

    def collect_data(self, test_y, pred_y, train_y, pred_y_train) -> None:
        labels = PRICE_BIN_LABELS
        self.avg_train_precs.append(
            precision_score(train_y, pred_y_train, average=self.avg_method, zero_division=0)
        )
        self.avg_train_recalls.append(
            recall_score(train_y, pred_y_train, average=self.avg_method, zero_division=0)
        )
        self.avg_train_f1s.append(
            f1_score(train_y, pred_y_train, average=self.avg_method, zero_division=0)
        )
        self.avg_test_precs.append(
            precision_score(test_y, pred_y, average=self.avg_method, zero_division=0)
        )
        self.avg_test_recalls.append(
            recall_score(test_y, pred_y, average=self.avg_method, zero_division=0)
        )
        self.avg_test_f1s.append(
            f1_score(test_y, pred_y, average=self.avg_method, zero_division=0)
        )
        self.per_class_train_precs.append(
            precision_score(train_y, pred_y_train, labels=labels, average=None, zero_division=0)
        )
        self.per_class_test_precs.append(
            precision_score(test_y, pred_y, labels=labels, average=None, zero_division=0)
        )

    def row(self, model: str) -> dict:
        row = {
            "model": model,
            "train_precision": self.avg_train_prec(),
            "train_recall": self.avg_train_recall(),
            "train_f1": self.avg_train_f1(),
            "test_precision": self.avg_test_prec(),
            "test_recall": self.avg_test_recall(),
            "test_f1": self.avg_test_f1(),
        }
        for i, value in enumerate(self.per_class_train_prec(), start=1):
            row[f"class_{i}_train_precision"] = value
        for i, value in enumerate(self.per_class_test_prec(), start=1):
            row[f"class_{i}_test_precision"] = value
        return row

    def avg_train_prec(self):
        return float(np.mean(self.avg_train_precs))

    def avg_train_recall(self):
        return float(np.mean(self.avg_train_recalls))

    def avg_train_f1(self):
        return float(np.mean(self.avg_train_f1s))

    def avg_test_prec(self):
        return float(np.mean(self.avg_test_precs))

    def avg_test_recall(self):
        return float(np.mean(self.avg_test_recalls))

    def avg_test_f1(self):
        return float(np.mean(self.avg_test_f1s))

    def per_class_train_prec(self):
        return np.mean(self.per_class_train_precs, axis=0)

    def per_class_test_prec(self):
        return np.mean(self.per_class_test_precs, axis=0)
