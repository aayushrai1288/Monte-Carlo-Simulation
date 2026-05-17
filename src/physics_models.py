import numpy as np

def initialize_ising_grid(n):
    """Creates an NxN grid with random spins (+1 or -1)."""
    return np.random.choice([1, -1], size=(n, n))

def calculate_energy_change(grid, i, j, n, j_interaction=1.0):
    """
    Calculates the change in energy if spin at (i, j) is flipped.
    Uses periodic boundary conditions.
    """
    spin = grid[i, j]
    neighbors = (
        grid[(i + 1) % n, j] +
        grid[(i - 1) % n, j] +
        grid[i, (j + 1) % n] +
        grid[i, (j - 1) % n]
    )
    # dE = -(-J * (-spin) * neighbors) - (-J * spin * neighbors)
    # dE = 2 * J * spin * neighbors
    return 2 * j_interaction * spin * neighbors

def run_ising_mcmc(grid, iterations, temperature, j_interaction=1.0):
    """
    Simulates the Ising model using the Metropolis algorithm.
    """
    n = grid.shape[0]
    current_grid = grid.copy()
    beta = 1.0 / temperature
    
    magnetization_history = []
    
    for _ in range(iterations):
        # Pick a random site
        i, j = np.random.randint(0, n, size=2)
        
        # Calculate energy change
        de = calculate_energy_change(current_grid, i, j, n, j_interaction)
        
        # Metropolis acceptance criterion
        if de <= 0 or np.random.rand() < np.exp(-de * beta):
            current_grid[i, j] *= -1
            
        magnetization_history.append(np.mean(current_grid))
        
    return current_grid, np.array(magnetization_history)
