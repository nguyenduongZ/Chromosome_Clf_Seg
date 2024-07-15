import torch
import numpy as np

def meanIOU_per_image(y_pred, y_true):
    # Chuyển đổi tensor sang numpy array
    y_pred_np = y_pred.detach().cpu().numpy()
    y_true_np = y_true.detach().cpu().numpy()
    
    # Chuyển đổi sang kiểu boolean
    y_pred_bool = (y_pred_np > 0.5).astype('bool')
    y_true_bool = (y_true_np > 0.5).astype('bool')

    # Tính toán IoU
    intersection = np.logical_and(y_pred_bool, y_true_bool).sum()
    union = np.logical_or(y_pred_bool, y_true_bool).sum()
    
    if union == 0:
        return 0.0  # tránh chia cho 0
    return intersection / union
