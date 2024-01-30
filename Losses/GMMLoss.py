import torch
from torch.distributions.normal import Normal as Gaussian

def gaussian_mixture_loss(batch, mus, sigmas, logpi):
    batch = batch.unsqueeze(-2)

    if isinstance(mus, list) or isinstance(mus, tuple):
        mus = torch.stack(mus)    
    if isinstance(sigmas, list) or isinstance(sigmas, tuple):
        sigmas = torch.stack(sigmas)
    if isinstance(logpi, list) or isinstance(logpi, tuple):
        logpi = torch.stack(logpi)    
 
    
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
    