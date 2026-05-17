import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.physics_models import initialize_ising_grid, run_ising_mcmc

def main():
    # Parameters
    N = 50
    ITERATIONS = 500000
    TEMPERATURE = 2.269 # Critical temperature approx
    
    print(f"Initializing {N}x{N} Ising grid...")
    grid = initialize_ising_grid(N)
    
    print(f"Running MCMC simulation at T={TEMPERATURE}...")
    final_grid, mag_history = run_ising_mcmc(grid, ITERATIONS, TEMPERATURE)
    
    print("Simulation complete.")
    
    # Plotting
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    ax[0].imshow(final_grid, cmap='binary')
    ax[0].set_title(f"Final Spin Configuration (T={TEMPERATURE})")
    
    ax[1].plot(mag_history, color='purple')
    ax[1].set_title("Magnetization History")
    ax[1].set_xlabel("Iteration")
    ax[1].set_ylabel("Average Magnetization")
    
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), 'ising_results.png')
    plt.savefig(output_path)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()
