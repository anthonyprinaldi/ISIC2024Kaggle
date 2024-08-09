import numpy as np
import torch
from sklearn.metrics import auc, roc_curve
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat


# FROM COMPETITION
def auc_20(y_true: np.ndarray, y_pred: torch.Tensor) -> float:
    v_gt = abs(y_true - 1)
    v_pred = -1.0 * y_pred.cpu().detach().numpy()
    min_tpr = 0.8
    max_fpr = abs(1-min_tpr)
    fpr, tpr, _ = roc_curve(v_gt, v_pred, sample_weight=None)
    if max_fpr is None or max_fpr == 1:
        auc_20 = auc(fpr, tpr)
    if max_fpr <= 0 or max_fpr > 1:
        raise ValueError("Expected min_tpr in range [0, 1), got: %r" % min_tpr)
        
    # Add a single point at max_fpr by linear interpolation
    stop = np.searchsorted(fpr, max_fpr, "right")
    x_interp = [fpr[stop - 1], fpr[stop]]
    y_interp = [tpr[stop - 1], tpr[stop]]
    tpr = np.append(tpr[:stop], np.interp(max_fpr, x_interp, y_interp))
    fpr = np.append(fpr[:stop], max_fpr)
    auc_20 = auc(fpr, tpr)

    return auc_20


class AUC_20(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("y_true", default=[], dist_reduce_fx="cat")
        self.add_state("y_pred", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        self.y_true.append(target.cpu().detach())
        self.y_pred.append(preds.cpu().detach())

    def compute(self) -> torch.Tensor:
        y_true = dim_zero_cat(self.y_true)
        y_pred = dim_zero_cat(self.y_pred)

        v_gt = abs(y_true.numpy() - 1)
        v_pred = -1.0 * y_pred.numpy()
        min_tpr = 0.8
        max_fpr = abs(1-min_tpr)
        fpr, tpr, _ = roc_curve(v_gt, v_pred, sample_weight=None)
        if max_fpr is None or max_fpr == 1:
            auc_20 = auc(fpr, tpr)
        if max_fpr <= 0 or max_fpr > 1:
            raise ValueError("Expected min_tpr in range [0, 1), got: %r" % min_tpr)
            
        # Add a single point at max_fpr by linear interpolation
        stop = np.searchsorted(fpr, max_fpr, "right")
        x_interp = [fpr[stop - 1], fpr[stop]]
        y_interp = [tpr[stop - 1], tpr[stop]]
        tpr = np.append(tpr[:stop], np.interp(max_fpr, x_interp, y_interp))
        fpr = np.append(fpr[:stop], max_fpr)
        auc_20 = auc(fpr, tpr)

        return torch.tensor(auc_20)