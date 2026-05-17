# MAMC Course Assignments - Aayush Rai

This repository contains a structured collection of assignments and solutions for the **MAMC (Markov Chain Monte Carlo)** course. It has been refactored into a modular Python project, moving logic from Jupyter notebooks into reusable scripts.

## 📂 Project Structure

```text
mamc/
├── assignments/            # Original Jupyter Notebooks and PDFs
│   ├── assignment_1/       # Introduction and Basic Concepts
│   ├── assignment_2/       # Sampling Methods and Theory
│   ├── ...
│   └── monte_carlo/        # Monte Carlo Integration
├── src/                    # Core modular logic
│   ├── samplers.py         # Metropolis & Metropolis-Hastings implementations
│   ├── physics_models.py    # Ising Model simulation logic
│   ├── stats_utils.py       # Burn-in, thinning, and ESS calculations
│   └── __init__.py
├── scripts/                # Standalone execution scripts
│   ├── run_gaussian_mcmc.py# Simulates a Gaussian distribution
│   └── run_ising_model.py  # Simulates the 2D Ising Model
├── requirements.txt        # Python dependencies
└── README.md
```

## 🛠️ Tech Stack & Requirements

- **Python 3.x**
- Core Libraries: `numpy`, `matplotlib`, `scipy`, `seaborn`

To install dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Running the Scripts

The logic from the assignments has been extracted into the `src/` directory for better reusability. You can run the example scripts from the root directory:

**Gaussian MCMC:**
```bash
python scripts/run_gaussian_mcmc.py
```

**Ising Model Simulation:**
```bash
python scripts/run_ising_model.py
```

## 📝 Assignments Overview

| Assignment | Focus Area | Key Topics |
|------------|------------|------------|
| 1 | Basics | Probability foundations and initial simulations. |
| 2 | Sampling | Introduction to various sampling techniques. |
| 3 | Markov Chains | Transition matrices, stationary distributions. |
| 4 | MCMC Intro | Theoretical grounding for MCMC algorithms. |
| 5 | MCMC Core | Implementation of Metropolis-Hastings or Gibbs Sampling. |
| 6 | Physics | 2D Ising Model simulation using Metropolis. |
| 7 | Final Work | Comprehensive application of MCMC methods. |

---
*Created and maintained by Aayush Rai.*
