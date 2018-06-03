import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve
from typing import Optional


class ClassificationResults:
    """
        Unified class for classification results.
    """
    def __init__(self,
                 labels: Optional[np.ndarray] = None,
                 preds: Optional[np.ndarray] = None,
                 pred_probs: Optional[np.ndarray] = None,
                 acc: Optional[float] = None,
                 loss: Optional[float] = None,
                 binary: bool = False):
        self.binary = binary
        self.labels = labels
        self.preds = preds
        self.pred_probs = pred_probs
        self.acc = acc
        self.loss = loss

    def get_auc_roc(self) -> Optional[float]:
        if not self.binary:
            return None
        return roc_auc_score(self.labels, self.pred_probs)

    def get_recall_for_precision(self, min_prec: float) -> Optional[float]:
        if not self.binary:
            return None
        precision, recall, thresholds = precision_recall_curve(self.labels, self.pred_probs)
        ret = 0.
        for i in range(len(thresholds)):  # = len(precision), = len(recall)
            if precision[i] >= min_prec:
                ret = max(ret, recall[i])
        return ret
