import torch


class DiffIs1Loss(torch.nn.Module):
    def forward(self, prediction, _):
        prediction = prediction.squeeze()
        mult_factor = 3
        diff = torch.abs((prediction[1:] - 1) - prediction[:-1])
        pow = torch.pow(diff, 2)
        loss = torch.mean(pow)
        return loss
