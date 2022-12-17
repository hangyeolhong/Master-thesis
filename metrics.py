import torch


def get_metrics(pred, original_data_y, ids):
    true_positive = torch.sum(torch.eq(pred[ids], original_data_y[ids]) * pred[ids])
    true_negative = torch.sum(torch.eq(pred[ids], original_data_y[ids]) * (1. - pred[ids]))
    false_positive = torch.sum(~torch.eq(pred[ids], original_data_y[ids]) * pred[ids])
    false_negative = torch.sum(~torch.eq(pred[ids], original_data_y[ids]) * (1. - pred[ids]))
    
    Precision = true_positive / (true_positive + false_positive)
    Recall = true_positive / (true_positive + false_negative)
    F1 = 2 * (Precision * Recall) / (Precision + Recall)

    return Precision, Recall, F1



def get_confusion_matrix(pred, original_data_y, ids):
    true_positive = torch.sum(torch.eq(pred[ids], original_data_y[ids]) * pred[ids])
    true_negative = torch.sum(torch.eq(pred[ids], original_data_y[ids]) * (1. - pred[ids]))
    false_positive = torch.sum(~torch.eq(pred[ids], original_data_y[ids]) * pred[ids])
    false_negative = torch.sum(~torch.eq(pred[ids], original_data_y[ids]) * (1. - pred[ids]))

    return true_positive, true_negative, false_positive, false_negative