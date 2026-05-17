import numpy as np

def metropolis_sampler(iterations, log_target_density, initial_val=0.0, proposal_width=1.0):
    """
    Runs a Metropolis sampler with a symmetric Gaussian proposal.
    
    Args:
        iterations (int): Number of steps to run the chain.
        log_target_density (callable): Function that returns the log-probability of a state.
        initial_val (float): Starting point of the chain.
        proposal_width (float): Standard deviation of the Gaussian proposal.
        
    Returns:
        tuple: (chain, acceptance_rate)
    """
    chain = [initial_val]
    accept_count = 0

    current_x = initial_val
    current_log_prob = log_target_density(current_x)

    for _ in range(iterations - 1):
        proposal_x = current_x + np.random.normal(scale=proposal_width)
        proposal_log_prob = log_target_density(proposal_x)

        log_alpha = proposal_log_prob - current_log_prob

        if np.log(np.random.rand()) < log_alpha:
            current_x = proposal_x
            current_log_prob = proposal_log_prob
            accept_count += 1

        chain.append(current_x)

    return np.array(chain), accept_count / iterations

def metropolis_hastings_sampler(iterations, log_target_density, proposal_func, log_proposal_ratio_func, initial_val=1.0):
    """
    Runs a general Metropolis-Hastings sampler.
    
    Args:
        iterations (int): Number of steps.
        log_target_density (callable): Target log-pdf.
        proposal_func (callable): Function to generate a new proposal from current state.
        log_proposal_ratio_func (callable): Function computing log(q(x|x') / q(x'|x)).
        initial_val (float): Initial state.
        
    Returns:
        tuple: (chain, acceptance_rate)
    """
    chain = np.zeros(iterations)
    chain[0] = initial_val
    accepted_moves = 0

    for i in range(1, iterations):
        current_x = chain[i-1]
        
        proposed_x = proposal_func(current_x)
        
        # log_alpha = log(p(x')/p(x)) + log(q(x|x')/q(x'|x))
        log_target_ratio = log_target_density(proposed_x) - log_target_density(current_x)
        log_proposal_ratio = log_proposal_ratio_func(current_x, proposed_x)
        
        log_alpha = log_target_ratio + log_proposal_ratio

        if np.log(np.random.rand()) < log_alpha:
            chain[i] = proposed_x
            accepted_moves += 1
        else:
            chain[i] = current_x

    return chain, accepted_moves / iterations
