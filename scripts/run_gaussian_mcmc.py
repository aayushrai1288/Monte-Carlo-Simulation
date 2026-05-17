import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.samplers import metropolis_sampler
from src.stats_utils import apply_burn_in

def main():
    # Parameters
    TARGET_MEAN = 5
    TARGET_STD = 1
    NUM_SAMPLES = 6000
    
    def log_target_density(x):
        return norm.logpdf(x, loc=TARGET_MEAN, scale=TARGET_STD)

    print(f"Running Metropolis Sampler for Gaussian({TARGET_MEAN}, {TARGET_STD})...")
    
    chain, acc_rate = metropolis_sampler(
        iterations=NUM_SAMPLES, 
        log_target_density=log_target_density,
        initial_val=0.0,
        proposal_width=1.0
    )
    
    clean_chain = apply_burn_in(chain, burn_in_pct=0.25)
    
    print(f"Acceptance Rate: {acc_rate:.2%}")
    print(f"Estimated Mean: {np.mean(clean_chain):.4f}")
    print(f"Estimated Std: {np.std(clean_chain):.4f}")
    
    # Plotting
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    ax[0].plot(chain, color='teal', alpha=0.7)
    ax[0].set_title("Trace Plot")
    
    ax[1].hist(clean_chain, bins=40, density=True, color='skyblue', alpha=0.6)
    x = np.linspace(TARGET_MEAN - 4*TARGET_STD, TARGET_MEAN + 4*TARGET_STD, 100)
    ax[1].plot(x, norm.pdf(x, TARGET_MEAN, TARGET_STD), 'r--', label='True PDF')
    ax[1].set_title("Posterior Distribution")
    ax[1].legend()
    
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), 'gaussian_mcmc_results.png')
    plt.savefig(output_path)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()
