# Calorimeter Image Compression with Convolutional Autoencoders

This repository contains a research notebook for training and evaluating a convolutional autoencoder for lossy compression of high-dimensional calorimeter image data. The project focuses on learning compact latent representations of detector events and evaluating whether the reconstructed images preserve downstream jet-level observables.

The workflow combines deep learning, numerical reconstruction metrics, scientific data preprocessing, and task-level validation using jet reconstruction.

## Overview

Each event is represented as a calorimeter image tensor with shape:

```text
(N_events, 50, 64, 6)
```

where the dimensions correspond to event number, eta bins, phi bins, and calorimeter channels.

The autoencoder maps each event into a compressed latent representation and reconstructs the original image from this bottleneck. The reconstruction is then assessed at two levels:

1. **Image-level reconstruction quality**, using reconstruction loss and error metrics.
2. **Jet-level preservation**, using anti-kt jet reconstruction, geometrical matching, and matched jet-pT response.

The central question is not only whether the model can minimise pixel-level error, but whether the compressed representation preserves information that remains useful after downstream reconstruction.

## Method

The implemented workflow is:

```text
Calorimeter image tensor
        ↓
Preprocessing
        ↓
Convolutional autoencoder
        ↓
Latent bottleneck representation
        ↓
Reconstructed calorimeter image
        ↓
Image-level reconstruction metrics
        ↓
Vectorisation into (pt, eta, phi, mass) objects
        ↓
FastJet anti-kt clustering
        ↓
Delta-R jet matching
        ↓
Matched jet-pT response analysis
```

## Autoencoder Model

The notebook implements a convolutional autoencoder in PyTorch. The encoder reduces the spatial-channel input into a lower-dimensional latent representation, while the decoder reconstructs the original calorimeter image from this compressed representation.

The model is trained using a reconstruction objective, with training and validation curves used to monitor convergence and generalisation. Bottleneck size can be varied to study the trade-off between compression strength and reconstruction quality.

## Jet-Level Validation

Pixel-level reconstruction loss is not sufficient on its own, because small image-level errors may still affect reconstructed physics objects. To test downstream preservation, the reconstructed calorimeter images are passed through the same jet reconstruction pipeline as the original images.

Each calorimeter image is converted into vector-like inputs:

```text
(pt, eta, phi, mass)
```

The pixel value is used as a pt-like weight, the eta and phi coordinates are assigned from the image grid, and the mass is set to zero.

Jets are reconstructed using the anti-kt algorithm with radius parameter:

```text
R = 0.4
```

Jets reconstructed from the original and reconstructed calorimeter images are then matched geometrically using:

```text
Delta R = sqrt((Delta eta)^2 + (Delta phi)^2)
```

For each matched jet pair, the relative transverse momentum error is calculated as:

```text
(pt_reco - pt_orig) / pt_orig
```

This measures the fractional change in reconstructed jet transverse momentum introduced by the compression and reconstruction pipeline.

## Outputs

The notebook produces:

- training and validation loss curves;
- reconstructed image comparisons;
- bottleneck-size comparison;
- reconstruction error summaries;
- anti-kt jet collections from original and reconstructed images;
- Delta-R matched jet pairs;
- matched jet-pT scatter plots;
- relative jet-pT error distributions;
- jet matching efficiency estimates.

## Technical Stack

- Python
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Awkward Array
- Scikit-HEP Vector
- FastJet
- Jupyter Notebook