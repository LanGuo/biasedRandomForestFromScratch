'''
Functions for evaluating classifier performance

Ref:
relevant functionalities in sklearn
'''

import numpy as np
from matplotlib import pyplot as plt


def binary_clf_curve(y_true, y_score, pos_label=None):
    '''
    Calculate true and false positives per binary classification threshold.
    Parameters
    ----------
    y_true : array, shape = [n_samples]
        True targets of binary classification
    y_score : array, shape = [n_samples]
        Estimated probabilities or decision function
    pos_label : int or str, default=None
        The label of the positive class
    Returns
    -------
    tps : array, shape = [n_thresholds <= len(np.unique(y_score))]
        An increasing count of true positives, at index i being the number
        of positive samples assigned a score >= thresholds[i].
    fns : array, shape = [n_thresholds]
        A count of false negatives, at index i being the number of positive
        samples assigned a score < thresholds[i].
    tns : array, shape = [n_thresholds]
        A count of true negatives, at index i being the number of negative
        samples assigned a score < thresholds[i].
    fps : array, shape = [n_thresholds]
        A count of false positives, at index i being the number of negative
        samples assigned a score >= thresholds[i].
    thresholds : array, shape = [n_thresholds]
        Decreasing score values.
    '''
    # convert y_true to indicate whether sample is in pos_label class
    y_true = (y_true == pos_label).astype(int)
    pos_num = sum(y_true)
    neg_num = len(y_true) - pos_num

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
    # accumulate the true positives with decreasing threshold
    tps = np.cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    fns = pos_num - tps
    tns = neg_num - fns
    return tps, fns, tns, fps, y_score[threshold_idxs]


def roc_curve(y_true, y_score, classes):
    '''
    Calculate Compute Receiver operating characteristic (ROC) and area under ROC curve.
    Parameters
    ----------
    y_true : array, shape = [n_samples, n_classes]
        True targets of binary classification
    y_score : array, shape = [n_samples, n_classes]
        Estimated probabilities or decision function
    classes : int or str, default=None
        The label of each class
    Returns
    -------
    tpr_classes : dict,
        tpr for each class
    fpr_classes : dict,
        fpr for each class
    roc_auc_classes : dict,
        roc auc for each class
    '''
    classes = np.unique(y_true)
    fpr_classes = dict()
    tpr_classes = dict()
    roc_auc_classes = dict()
    for i,c in enumerate(classes):
        tps, _, _, fps, _ = binary_clf_curve(y_true, y_score[:, i], pos_label=c)
        # ensure that the curve starts at (0, 0)
        tps = np.r_[0, tps]
        fps = np.r_[0, fps]
        tpr = tps / tps[-1]
        fpr = fps / fps[-1]
        fpr_classes[c] = fpr
        tpr_classes[c] = tpr
        roc_auc_classes[c] = auc(fpr, tpr)
    return tpr_classes, fpr_classes, roc_auc_classes


def prc_curve(y_true, y_score, classes):
    '''
    Calculate .
    Parameters
    ----------
    y_true : array, shape = [n_samples, n_classes]
        True targets of binary classification
    y_score : array, shape = [n_samples, n_classes]
        Estimated probabilities or decision function
    classes : int or str, default=None
        The label of each class
    Returns
    -------
    tpr_classes : dict,
        tpr for each class
    fpr_classes : dict,
        fpr for each class
    prc_auc_classes : dict,
        prc auc for each class
    '''
    classes = np.unique(y_true)
    precision_classes = dict()
    recall_classes = dict()
    prc_auc_classes = dict()
    for i,c in enumerate(classes):
        tps, _, _, fps, _ = binary_clf_curve(y_true, y_score[:, i], pos_label=i)
        precision = tps / (tps + fps)
        precision[np.isnan(precision)] = 0
        recall = tps / tps[-1]
        # stop when full recall attained
        # and reverse the outputs so recall is decreasing
        last_ind = tps.searchsorted(tps[-1])
        sl = slice(last_ind, None, -1)
        precision = np.r_[precision[sl], 1]
        recall = np.r_[recall[sl], 0]
        precision_classes[c] = precision
        recall_classes[c] = recall
        prc_auc_classes[c] = auc(recall, precision)
    return precision_classes, recall_classes, prc_auc_classes


def plot_roc_curve(fpr, tpr, class_label, **kwargs):
    fig = plt.figure()
    plt.plot(fpr, tpr, **kwargs)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC curve for class {class_label}')
    plt.legend()
    return fig


def plot_prc_curve(recall, precision, class_label, **kwargs):
    fig = plt.figure()
    plt.plot(recall, precision, **kwargs)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.title(f'Precision-recall curve for class {class_label}')
    return fig


def auc(x, y):
    '''
    Compute Area Under the Curve (AUC) using the trapezoidal rule
    Parameters
    ----------
    x : array, shape = [n]
        x coordinates. These must be either monotonic increasing or monotonic
        decreasing.
    y : array, shape = [n]
        y coordinates.
    Returns
    -------
    auc : float
    '''
    direction = 1
    dx = np.diff(x)
    if np.any(dx < 0):
        if np.all(dx <= 0):
            direction = -1
        else:
            raise ValueError("x is neither increasing nor decreasing "
                             ": {}.".format(x))

    area = direction * np.trapz(y, x)
    if isinstance(area, np.memmap):
        # Reductions such as .sum used internally in np.trapz do not return a
        # scalar by default for numpy.memmap instances contrary to
        # regular numpy.ndarray instances.
        area = area.dtype.type(area)
    return area


def confusion_matrx_by_class(y_true, y_pred):
    '''
    Compute a confusion matrix for the specific (positive) class for binary classification.
    Parameters
    ----------
    y_true : array, shape = [n_samples]
        True targets of binary classification
    y_pred : array, shape = [n_samples]
        Estimated probabilities or decision function
    Returns
    -------
    confusion : array, shape (n_classes, 2, 2)
        A 2x2 confusion matrix [[tn, fp], [fn, tp]] corresponding to each output in the input.

    '''
    assert (y_true.size == y_pred.size) and (y_true.size > 1)
    classes = np.unique(y_true)
    confusion_mats = dict()
    for c in classes:
        tp = sum(y_pred[y_true==c] == c)
        fn = sum(y_pred[y_true==c] != c)
        tn = sum(y_pred[y_true!=c] != c)
        fp = sum(y_pred[y_true!=c] == c)
        confusion_mats[c] = np.array([tn, fp, fn, tp]).reshape(2, 2)
    return confusion_mats


def precision_recall_score_by_class(y_true, y_pred):
    '''
    Compute precision recall scores for the specific class for binary classification.
    Parameters
    ----------
    y_true : array, shape = [n_samples]
        True targets of binary classification
    y_pred : array, shape = [n_samples]
        Estimated probabilities or decision function
    Returns
    -------
    precision : float, shape = [n_class]
    recall : float, shape = [n_class]
    '''
    classes = np.unique(y_true)
    precision_by_class = dict()
    recall_by_class = dict()
    confusion_mats = confusion_matrx_by_class(y_true, y_pred)
    for i in classes:
        [[tn, fp], [fn, tp]] = confusion_mats[i]
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        precision_by_class[i] = precision
        recall_by_class[i] = recall
    return precision_by_class, recall_by_class

