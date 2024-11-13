import torch as tr

def _pairwise_distances(points):
    X = points
    sum_X = tr.sum(X**2, dim=1).view(-1, 1)
    D = sum_X + sum_X.T - 2 * tr.mm(X, X.T)
    return D

def reduce_sne(X, dim_low, num_iters=1000):
    num_samples = X.shape[0]
    nondiagmask = (1.0 - tr.eye(num_samples))

    def prob_matrix(distances, beta=1):
        probs = tr.exp(-distances * beta)
        probs = probs * nondiagmask
        probs_sum = probs.sum(dim=1, keepdim=True)
        probs = probs / probs_sum
        return probs

    D = _pairwise_distances(X)
    P = prob_matrix(D, beta=1.0 / (tr.mean(D) + 1e-8))

    # Initialize random points
    R = tr.randn((num_samples, dim_low), requires_grad=True)

    # Initialize optimizer
    optimizer = tr.optim.Adam([R], lr=0.1)

    for i in range(num_iters):
        # clear gradients
        optimizer.zero_grad()

        # Calculate KL divergence
        rD = _pairwise_distances(R)
        Q = prob_matrix(rD, 1)
        KLD = tr.sum(P * tr.log((P + 1e-8) / (Q + 1e-8)))

        # Backpropagate
        KLD.backward()

        # Step optim
        optimizer.step()

    return R.detach()