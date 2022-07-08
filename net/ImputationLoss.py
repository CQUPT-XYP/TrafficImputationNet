import torch
import torch.nn as nn
import numpy as np

class ImputationLoss(nn.Module):
    def __init__(self):
        super(ImputationLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        

    def forward(self, pred, target, batch_loss_indices, loss_matrix):
        pred_result = pred * loss_matrix
        target_result = target * loss_matrix
        pred_result = pred_result[loss_matrix != 0]
        target_result = target_result[loss_matrix != 0]

        loss = torch.sqrt(self.mse_loss(pred_result, target_result))

        return loss