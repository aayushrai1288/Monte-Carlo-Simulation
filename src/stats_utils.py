import numpy as np

def apply_burn_in(chain, burn_in_pct=0.2):
    """
    Removes the burn-in period from an MCMC chain.
    """
    n = len(chain)
    burn_idx = int(n * burn_in_pct)
    return chain[burn_idx:]

def apply_thinning(chain, thinning_factor=10):
    """
    Thins the chain to reduce autocorrelation.
    """
    return chain[::thinning_factor]

def calculate_ess(chain):
    """
    Rough estimate of Effective Sample Size (ESS).
    """
    # Simplified version for demonstration
    from scipy.stats import norm
    n = len(chain)
    if n < 2: return n
    
    # Using statsmodels or similar would be better, but keeping it simple
    # auto-correlation at lag 1
    mu = np.mean(chain)
    var = np.var(chain)
    if var == 0: return n
    
    rho1 = np.corrcoef(chain[:-1], chain[1:])[0, 1]
    ess = n * (1 - rho1) / (1 + rho1)
    return ess
