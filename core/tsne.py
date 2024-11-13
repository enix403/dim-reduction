import torch as tr
import torch.nn.functional as F

def _pairwise_distances(points):
    X = points
    sum_X = tr.sum(X**2, dim=1).view(-1, 1)
    D = sum_X + sum_X.T - 2 * tr.mm(X, X.T)
    return D

def _compute_high_dim_affinities(X, perplexity):
    # Compute pairwise Euclidean distances D
    distances = _pairwise_distances(X)

    num_samples = X.shape[0]
    
    # Initialize beta (precision of Gaussian distribution)
    beta = tr.ones(num_samples, 1)
    
    # Init with zeros It will be filled later with binary
    # search to get P values based on the perplexity target
    P = tr.zeros_like(distances)
    
    # The target entropy is the log2 of perplexity
    log_perplexity = tr.log(tr.tensor(perplexity))
    H_target = log_perplexity
    
    for i in range(num_samples):
        # Track min/max precisions to prevent making
        # betas worse than previously explored values
        betamin, betamax = None, None

        # Get distances to all points, excluding self-distance
        select_idx = tr.arange(num_samples) != i
        Di = distances[i, select_idx]
        
        # Perform binary search on beta to match the target perplexity
        for _ in range(50):
            # Gaussian affinities for point i
            Pi = tr.exp(-Di * beta[i])
            # Normalize
            Pi = Pi / tr.sum(Pi)

            # Entropy of Pi
            H = -tr.sum(Pi * tr.log(Pi))
            
            Hdiff = H_target - H

            if Hdiff > 0:
                # Need more entropy, increase variance (decrease B)
                betamax = beta[i].clone()
                if betamin is None:
                    beta[i] = beta[i] / 2
                else:
                    beta[i] = (beta[i] + betamin) / 2
            else:
                # Need less entropy, decrease variance (increase B)
                betamin = beta[i].clone()
                if betamax is None:
                    beta[i] = beta[i] / 2
                else:
                    beta[i] = (beta[i] + betamax) / 2
            
            if tr.abs(Hdiff) < 1e-4:
                break
        
        P[i, select_idx] = Pi
    
    return P


def _kl_divergence_loss(P, Q, eps=1e-8):
    return tr.sum(P * tr.log((P + eps) / (Q + eps)))

def reduce_tsne(
    X,
    dim_low,
    perplexity=50.0,
    n_iter=1000,
    lr=0.5,
    annealing_factor=4,
    annealing_steps=100
):
    num_samples = X.shape[0]

    # Compute high dimensional probabilities
    P = _compute_high_dim_affinities(X, perplexity)

    # Symmetrize P
    P = (P + P.T) / (2.0 * tr.sum(P))

    # Early exaggeration
    P_exg = P * annealing_factor

    # Initialize low-dimensional map Y randomly
    R = tr.randn(num_samples, dim_low, requires_grad=True)

    optimizer = tr.optim.Adam([R], lr=lr)

    nondiagmask = (1.0 - tr.eye(num_samples))

    for i in range(n_iter):
        optimizer.zero_grad()

        # Computer low dimensional pairwise distances
        Dr = _pairwise_distances(R)

        # Apply t-distribution similarities
        Q = 1 / (1 + Dr)

        # Set diagonal to zero
        Q = Q * nondiagmask

        # Normalize
        Q = Q / tr.sum(Q)

        # Compute KL divergence loss
        loss = _kl_divergence_loss(P_exg, Q)
        
        # Perform optimization step
        loss.backward()
        optimizer.step()

        # Anneal P to remove early exaggeration
        if i == annealing_steps:
            P_exg = P

    return R.detach()
