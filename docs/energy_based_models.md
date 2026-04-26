# Energy-Based Models

Energy-based models define an unnormalized density with a scalar energy function. Low-energy regions are more likely, but the normalizing constant is usually hard to compute.

The toy experiment in this repository uses an analytic Gaussian-mixture energy so the focus can stay on the sampling behavior:

- `energy(x)` assigns lower values near mixture centers.
- `score(x)` gives the gradient of the log density.
- `langevin_sample(...)` repeatedly follows the score and injects Gaussian noise.

This setup is useful for learning because every part is inspectable. A neural EBM would replace the analytic energy with a learned network, while keeping the same sampling idea.
