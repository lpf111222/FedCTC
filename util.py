import torch
import random
import numpy as np

def set_ramdom_seed(seed=2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.synchronize()
        torch.backends.cudnn.deterministic = True  
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.initial_seed = seed
    return None

def roc_curve(y_true, y_scores):
    # The data type of the input  data
    y_true = np.array(y_true, dtype=np.bool)
    y_score = np.array(y_scores, dtype=np.float64)
    # index
    pos_indexs = y_true
    neg_indexs = ~pos_indexs
    # Positive and negative samples
    pos_scores = y_score[pos_indexs]
    neg_scores = y_score[neg_indexs]
    # Between 0 and 1, with an interval of 0.01
    thresholds = np.linspace(1.00, 0, 101).round(2)
    # descending
    pos_scores_descending = np.sort(pos_scores)[::-1]
    neg_scores_descending = np.sort(neg_scores)[::-1]
    # total
    total_pos = pos_scores_descending.size
    total_neg = neg_scores_descending.size
    # Calculate FPR and TPR
    fpr_list = []
    tpr_list = []
    for threshold in thresholds:
        # TPR
        tp_count = np.sum(pos_scores_descending >= threshold)
        tpr = tp_count / total_pos if total_pos != 0 else 0.0
        tpr_list.append(tpr)
        # FPR
        fp_count = np.sum(neg_scores_descending >= threshold)
        fpr = fp_count / total_neg if total_neg != 0 else 0.0
        fpr_list.append(fpr)
    # return sklearn.metrics.roc_curve(y_true, y_scores)
    return np.array(fpr_list), np.array(tpr_list), thresholds

def precision_recall_curve(y_true, y_scores):
    # The data type of the input data
    y_true = np.array(y_true, dtype=np.bool)
    y_score = np.array(y_scores, dtype=np.float64)
    # index
    pos_indexs = y_true
    neg_indexs = ~pos_indexs
    # Positive and negative samples
    pos_scores = y_score[pos_indexs]
    neg_scores = y_score[neg_indexs]
    # Between 0 and 1, with an interval of 0.01
    thresholds = np.linspace(1.00, 0, 101).round(2)
    # descending
    pos_scores_descending = np.sort(pos_scores)[::-1]
    neg_scores_descending = np.sort(neg_scores)[::-1]
    # total
    total_pos = pos_scores_descending.size
    total_neg = neg_scores_descending.size
    # Calculate Precision and Recall
    precision_list = []
    recall_list = []
    for threshold in thresholds:
        # Recall
        tp_count = np.sum(pos_scores_descending >= threshold)
        recall = tp_count / total_pos if total_pos != 0 else 0.0
        recall_list.append(recall)
        # precision
        fp_count = np.sum(neg_scores >= threshold)
        if tp_count + fp_count == 0:
            precision = 1.0
        else:
            precision = tp_count / (tp_count + fp_count)
        precision_list.append(precision)
    return np.array(precision_list), np.array(recall_list), thresholds