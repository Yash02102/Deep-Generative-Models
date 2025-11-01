# β-VAE Experiment Summary (MNIST)

## Environment note
Training runs that depend on `torch` could not be executed inside the evaluation container because PyTorch is not available and downloading the dependency through the proxied network was blocked. Consequently, the comparisons below are qualitative, extrapolating the expected behaviour of the provided `ConvVAE` and `train_conv_vae` routines in `VAE_MNIST.ipynb` under multiple β settings.

## Qualitative comparison across β settings
| β value | Reconstruction loss (relative) | KL divergence (relative) | Reconstruction quality | Generation quality |
|:------:|:------------------------------:|:------------------------:|:-----------------------|:-------------------|
| 0.5    | lowest                         | lowest                   | Sharp digits with high fidelity | Samples resemble training data closely but offer limited variation |
| 1.0    | low                            | moderate                 | Slightly blurrier yet still faithful digits | Improved diversity with realistic strokes |
| 3.0    | medium                         | higher                   | Noticeable blurring and occasional digit ambiguity | Broader latent usage leading to more varied but less crisp samples |
| 5.0    | highest                        | highest                  | Reconstructions lose fine details and may misclassify digits | Highly diverse generations, yet realism drops because samples become washed out |

The "relative" columns are ranked comparisons: higher β monotonically increases the contribution of the KL term, forcing encoder posteriors to remain close to the unit Gaussian. This inflates the measured KL divergence while preventing the model from allocating capacity to precise reconstructions, so the reconstruction loss grows with β.

## Effect on latent representations
Increasing β strengthens the pressure for each latent dimension to match the prior, promoting factorised posteriors. When β ≈ 1 (the conventional VAE), the model balances reconstruction fidelity and latent regularisation, producing latent codes that capture a mixture of shared and disentangled factors. As β is raised (3–5), the KL penalty dominates: encodings become more isotropic and independent, which encourages disentanglement and yields a latent space where traversals manipulate interpretable attributes (stroke thickness, slant, etc.). However, the stronger regularisation also restricts the amount of information each latent dimension can carry, so the decoder receives a less detailed signal and reconstructions appear blurrier.

Conversely, lowering β below 1 relaxes the information bottleneck. The encoder can devote more capacity to per-sample details, restoring sharp reconstructions but collapsing the latent space around training samples, which harms the model's ability to generate diverse digits.

## Takeaways
- β provides a knob between reconstruction fidelity and latent factorisation.
- Larger β values promote disentanglement by encouraging isotropic posteriors, but the resulting information bottleneck makes reconstructions less crisp.
- Moderate β (around 1) offers a compromise between faithful reconstructions and useful, moderately disentangled latent dimensions.
