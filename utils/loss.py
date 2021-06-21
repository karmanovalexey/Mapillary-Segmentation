import torch

from utils.mapillary_pallete import MAPILLARY_LOSS_WEIGHTS

NUM_CLASSES = 66

class FocalLoss(torch.nn.Module):
    """
    based on https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
    """
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = torch.nn.functional.log_softmax(input,-1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.data.exp()

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

class Loss(torch.nn.Module):
    def __init__(self, args, ocr_coeff = 0.4):
        super().__init__()
        self.model_name = args.model
        if args.loss=='BCE':
            loss_weights = torch.Tensor(MAPILLARY_LOSS_WEIGHTS).to(device=args.device)
            self.loss = torch.nn.CrossEntropyLoss(weight=loss_weights) 
        elif args.loss=='Focal':
            self.loss = FocalLoss()
        self.k = ocr_coeff

    def forward(self, outputs, targets):
        if self.model_name == 'resnet_ocr':
            (out_aux, out) = outputs
            aux_loss = self.loss(out_aux, targets)
            out_loss = self.loss(out, targets)
            return self.k*aux_loss + out_loss
        else:
            return self.loss(outputs, targets)
