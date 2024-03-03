import torch
from torch.distributions.normal import Normal as Gaussian
import torch.nn as nn
import torch.nn.functional as F

def reduction_losses(losses, reduction): 
    if reduction == 'mean':
        return torch.mean(torch.stack(losses))
    if reduction == 'sum':
        return torch.sum(torch.stack(losses))
    return losses


def gaussian_mixture_loss(batch, mus, sigmas, logpi):

    if isinstance(batch, nn.utils.rnn.PackedSequence):
        batch = nn.utils.rnn.unpack_sequence(batch)

    if isinstance(batch, list) or isinstance(batch, tuple):
        batch = torch.stack(batch)    
    if isinstance(mus, list) or isinstance(mus, tuple):
        mus = torch.stack(mus)    
    if isinstance(sigmas, list) or isinstance(sigmas, tuple):
        sigmas = torch.stack(sigmas)
    if isinstance(logpi, list) or isinstance(logpi, tuple):
        logpi = torch.stack(logpi)    
    

    #losses = [_gaussian_mixture_loss(*v) for v in zip(batch, mus, sigmas, logpi)]
    losses = _gaussian_mixture_loss(batch, mus, sigmas, logpi)
    
    return losses

def _gaussian_mixture_loss(batch, mus, sigmas, logpi):
    
    batch = batch.swapaxes(0, 1)
    mus = mus.swapaxes(0, 1)
    sigmas = sigmas.swapaxes(0, 1)
    logpi = logpi.swapaxes(0, 1)

    batch = batch.unsqueeze(-2)
    # Inititiate the Gaussian Dists
    normal_dist = Gaussian(mus, sigmas)
    # Get the Log(P(x_i,k)~N_{mus, sigma}) => (batch, gaussian, feature)
    g_log_probs = normal_dist.log_prob(batch)
    # Reduce the feature dim and add bias:
    #   log(pi_i) + (log(x_i,1) + log(x_i,2) + ... + log(x_i,n)) => log(pi_i * x_i)
    g_log_probs = logpi + torch.sum(g_log_probs, dim=-1)
    # Get the most probable dist:
    #   max_i log(pi_i * x_i)
    max_log_probs = torch.max(g_log_probs, dim=-1, keepdim=True)[0]
    # log(pi_i * x_i) - max_i log(pi_i * x_i)
    g_log_probs = g_log_probs - max_log_probs

    # exp(log(pi_i * x_i) - max_i log(pi_i * x_i))
    # exp(log(pi_i * x_i)) / exp(max_i log(pi_i * x_i)))
    # (pi_i * x_i) / max_i (pi_i * x_i)
    # (1 / max_i (pi_i * x_i)) * (pi_i * x_i)
    g_probs = torch.exp(g_log_probs)
    # Sum_i ( (1 / max_i (pi_i * x_i)) * (pi_i * x_i) )
    # (1 / max_i (pi_i * x_i)) * Sum_i (pi_i * x_i)
    probs = torch.sum(g_probs, dim=-1)

    # max_i log(pi_i * x_i) + log( (1 / max_i (pi_i * x_i)) * Sum_i (pi_i * x_i) )
    log_prob = max_log_probs.squeeze() + torch.log(probs)
    
    
    return -1 * torch.mean(log_prob)
    

def bce_with_logits_list(y_pred, y_true, reduction = 'mean'):

    y_pred = [y_.float() for y_ in y_pred]
    y_true = [y.float() for y in y_true]

    losses = [F.binary_cross_entropy_with_logits(y_, y, reduction=reduction) for y_, y in zip(y_pred, y_true)]
    
    return reduction_losses(losses, reduction)



def mse_loss_list(y_pred, y_true, reduction = 'mean'):

    y_pred = [y_.float() for y_ in y_pred]
    y_true = [y.float() for y in y_true]

    losses = [F.mse_loss(y_, y, reduction=reduction) for y_, y in zip(y_pred, y_true)]
    
    return reduction_losses(losses, reduction)

