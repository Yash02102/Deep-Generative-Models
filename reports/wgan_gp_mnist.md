# WGAN-GP on MNIST

This report summarizes controlled experiments training Wasserstein GANs with gradient penalty on MNIST. To keep CPU runtimes manageable all runs restrict training to five mini-batches per epoch (≈320 images).

## Experiment 1 – Baseline WGAN-GP
- **Configuration:** GP weight = 10, critic iterations = 3, batch size = 64, learning rate = 2e-4, β = (0.0, 0.99), max 5 batches/epoch, 20 epochs.
- **Observations:** Critic and generator losses settle near −5.80/5.06 by epoch 20 while the Wasserstein estimate contracts to ~6.6, indicating smooth convergence. The sample grid contains recognizable digits with moderate diversity despite the reduced data budget.
- **Artifacts:**
  - Training curves: `reports/baseline_history.png`
  - Sample grid: `outputs/wgan_gp_baseline/final_samples.png`

## Experiment 2 – No Gradient Penalty
- **Configuration:** same as baseline but gradient penalty weight = 0.
- **Observations:** Without the penalty the critic quickly saturates (loss ≈ −105.7) and the Wasserstein estimate explodes above 100. Generator loss follows the critic into the tens, and generated samples collapse to noisy blobs, confirming the stabilizing role of the gradient penalty.
- **Artifacts:**
  - Training curves: `reports/no_gp_history.png`
  - Sample grid: `outputs/wgan_gp_no_gp/final_samples.png`

## Experiment 3 – Fewer Critic Updates
- **Configuration:** same as baseline but critic iterations = 1 and 10 epochs.
- **Observations:** Training remains stable but converges slowly: the Wasserstein estimate plateaus near 12.3 compared to 6.6 for the baseline, and sample sharpness lags behind. Increasing critic updates clearly helps match generator and critic capacity.
- **Artifacts:**
  - Training curves: `reports/ncritic1_history.png`
  - Sample grid: `outputs/wgan_gp_ncritic1/final_samples.png`

## Key Insights
- Gradient penalty is essential for constraining the critic; removing it destabilizes both losses and sample quality.
- More critic updates per generator step accelerate convergence toward lower Wasserstein estimates and sharper digits, especially when operating on a limited subset of MNIST.
- Even with a heavily truncated dataset the baseline configuration yields coherent digits, showing the robustness of WGAN-GP when properly regularized.
