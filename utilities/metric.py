from medpy.metric import hd95
import numpy as np


def compute_BraTS_HD95(ref, pred, spacing = (3.600, 0.625, 0.625)):
    """
    ref and gt are binary integer numpy.ndarray s
    spacing is assumed to be (1, 1, 1)
    :param ref:
    :param pred:
    :return:
    """
    num_ref = np.sum(ref)
    num_pred = np.sum(pred)

    if num_ref == 0:
        if num_pred == 0:
            return 0
        else:
            return 373.12866
    elif num_pred == 0 and num_ref != 0:
        return 373.12866
    else:
        pred_hd = np.squeeze(pred, axis=0)
        ref_hd = np.squeeze(ref, axis=0)
        return hd95(pred_hd, ref_hd, spacing)
