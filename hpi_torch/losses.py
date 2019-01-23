#encoding: utf-8
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

from torch.autograd import Variable, Function
import torch


class FocalLossWithLogits(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + \
               ((-max_val).exp() + (-input - max_val).exp()).log()

        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss

        return loss.sum(dim=1).mean()


class BalanceLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(BalanceLoss, self).__init__()

    def forward(self, x, y):
        x = x.sigmoid()
        loss = (1.0 - x) * x
        loss = loss.mean()

        return loss


def _to_one_hot(y, n_dims=None):
    """
    Take integer y (tensor or variable) with n dims and
    convert it to 1-hot representation with n+1 dims
    """
    y_tensor = y.data if isinstance(y, Variable) else y
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(y.size()[0], -1)

    return Variable(y_one_hot) if isinstance(y, Variable) else y_one_hot


class WARP(Function):
    """Weighted Approximate-Rank Pairwise loss
    Autograd function of WARP loss. Appropirate for multi-label
    - Reference:
      https://medium.com/@gabrieltseng/intro-to-warp-loss-automatic-differentiation-and-pytorch-b6aa5083187a
    """
    @staticmethod
    def forward(ctx, input, target, max_num_trials=None):
        batch_size = target.size()[0]
        label_size = target.size()[1]
        ## rank weight
        rank_weights = [1.0 / 1]
        for i in range(1, label_size):
            rank_weights.append(rank_weights[i - 1] + (1.0 / i + 1))

        if max_num_trials is None:
            max_num_trials = target.size()[1] - 1

        positive_indices = target.gt(0).float()
        negative_indices = target.eq(0).float()
        L = torch.zeros(input.size())
        for i in range(batch_size):
            for j in range(label_size):
                if target[i, j] == 1:
                    ## initialization
                    sample_score_margin = -1
                    num_trials = 0
                    while ((sample_score_margin < 0) and (num_trials < max_num_trials)):
                        ## sample a negative label, to only determine L (ranking weight)
                        neg_labels_idx = np.array([idx for idx, v in enumerate(target[i, :]) if v == 0])
                        if len(neg_labels_idx) > 0:
                            neg_idx = np.random.choice(neg_labels_idx, replace=False)
                            ## if models.py thinks neg ranks before pos...
                            sample_score_margin = input[i, neg_idx] - input[i, j]
                            num_trials += 1

                        else:  # ignore cases where all labels are 1...
                            num_trials = 1
                            pass
                    ## how many trials determine the weight
                    r_j = int(np.floor(max_num_trials / num_trials))
                    L[i, j] = rank_weights[r_j]

        ## summing over all negatives and positives
        # -- since inputs are sigmoided, no need for clamp with min=0
        loss = torch.sum(L * (torch.sum(1 - positive_indices * input + negative_indices * input, dim=1, keepdim=True)), dim=1)
        # ctx.save_for_backward(input, target)
        ctx.L = L
        ctx.positive_indices = positive_indices
        ctx.negative_indices = negative_indices

        return torch.sum(loss, dim=0)

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # input, target = ctx.saved_variables
        L = Variable(ctx.L, requires_grad=False)
        positive_indices = Variable(ctx.positive_indices, requires_grad=False)
        negative_indices = Variable(ctx.negative_indices, requires_grad=False)
        pos_grad = torch.sum(L, dim=1, keepdim=True) * (-positive_indices)
        neg_grad = torch.sum(L, dim=1, keepdim=True) * negative_indices
        grad_input = grad_output * (pos_grad + neg_grad)

        return grad_input, None, None

class WARPLoss(nn.Module):
    def __init__(self, max_num_trials=None):
        super(WARPLoss, self).__init__()
        self.max_num_trials = max_num_trials

    def forward(self, input, target):
        return WARP.apply(input.cpu(), target.cpu(), self.max_num_trials)


class LSEP(Function):
    """
    Autograd function of LSEP loss. Appropirate for multi-label
    - Reference: Li+2017
      https://arxiv.org/pdf/1704.03135.pdf
    """
    @staticmethod
    def forward(ctx, input, target, max_num_trials=None):
        batch_size = target.size()[0]
        label_size = target.size()[1]

        ## rank weight
        rank_weights = [1.0 / 1]
        for i in range(1, label_size):
            rank_weights.append(rank_weights[i - 1] + (1.0 / i + 1))

        if max_num_trials is None:
            max_num_trials = target.size()[1] - 1

        positive_indices = target.gt(0).float()
        negative_indices = target.eq(0).float()

        ## summing over all negatives and positives
        loss = 0.
        for i in range(input.size()[0]):  # loop over examples
            pos = np.array([j for j, pos in enumerate(positive_indices[i]) if pos != 0])
            neg = np.array([j for j, neg in enumerate(negative_indices[i]) if neg != 0])

            for j, pj in enumerate(pos):
                for k, nj in enumerate(neg):
                    loss += np.exp(input[i, nj] - input[i, pj])

        loss = torch.from_numpy(np.array([np.log(1 + loss)])).float()
        ctx.save_for_backward(input, target)
        ctx.loss = loss
        ctx.positive_indices = positive_indices
        ctx.negative_indices = negative_indices

        return loss

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        input, target = ctx.saved_variables
        loss = Variable(ctx.loss, requires_grad=False)
        positive_indices = ctx.positive_indices
        negative_indices = ctx.negative_indices

        fac = -1 / loss
        grad_input = torch.zeros(input.size())

        ## make one-hot vectors
        one_hot_pos, one_hot_neg = [], []

        for i in range(grad_input.size()[0]):  # loop over examples
            pos_ind = np.array([j for j, pos in enumerate(positive_indices[i]) if pos != 0])
            neg_ind = np.array([j for j, neg in enumerate(negative_indices[i]) if neg != 0])
            one_hot_pos.append(_to_one_hot(torch.from_numpy(pos_ind), input.size()[1]))
            one_hot_neg.append(_to_one_hot(torch.from_numpy(neg_ind), input.size()[1]))

        ## grad
        for i in range(grad_input.size()[0]):
            for dum_j, phot in enumerate(one_hot_pos[i]):
                for dum_k, nhot in enumerate(one_hot_neg[i]):
                    grad_input[i] += (phot - nhot) * torch.exp(-input[i].data * (phot - nhot))
        ##
        grad_input = Variable(grad_input) * (grad_output * fac)

        return grad_input, None, None


class LSEPLoss(nn.Module):
    def __init__(self):
        super(LSEPLoss, self).__init__()

    def forward(self, input, target):
        return LSEP.apply(input.cpu(), target.cpu())


def loss_lsep(outputs, labels):
    """Log-sum-exp-pairwise loss
    Sigmoid + LSEP loss
    """
    return LSEPLoss()(F.sigmoid(outputs), labels)


def loss_warp(outputs, labels):
    """
    Sigmoid + WARP loss
    """
    return WARPLoss()(F.sigmoid(outputs), labels)


def get_loss():
    losses = {}
    losses['bce'] = nn.BCEWithLogitsLoss()
    losses['focal'] = FocalLossWithLogits()
    losses['warp'] = loss_warp
    losses['lsep'] = loss_lsep

    return losses