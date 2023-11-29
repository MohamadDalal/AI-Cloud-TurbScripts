#import numpy as np
import torch
import torch.nn as nn


class div_loss(nn.Module):
    def __init__(self):
        super(div_loss, self).__init__()


    def calc_div(self, data):
        x_grad_x, x_grad_y = torch.gradient(data[:, 0, :, :], dim=(1, 2))
        y_grad_x, y_grad_y = torch.gradient(data[:, 1, :, :], dim=(1, 2))
        z_grad_x, z_grad_y = torch.gradient(data[:, 2, :, :], dim=(1, 2))

        divergence_x = x_grad_x + x_grad_y
        divergence_y = y_grad_x + y_grad_y
        divergence_z = z_grad_x + z_grad_y

        tot_div_x = torch.sum(divergence_x)
        tot_div_y = torch.sum(divergence_y)
        tot_div_z = torch.sum(divergence_z)

        return (tot_div_x + tot_div_y + tot_div_z)/3
        

    def forward(self, predictions, targets):
        pred_div = self.calc_div(predictions)
        target_div = self.calc_div(targets)
        return abs(pred_div - target_div)