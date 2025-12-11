# Closed-Set Device Classification

Device classification using channel-state information (CSI) from the 5G physical uplink shared channel (PUSCH) with radio frequency fingerprinting (RFFI).

This repository implements a neural network-based system for closed-set device classification using CSI features extracted from 5G PUSCH signals. The method uses CSI obfuscation-inspired features to extract device-specific RF fingerprints that are independent of the wireless channel and transmitter location.

## Overview

This implementation trains a neural network to identify a user equipment (UE) within a closed set of candidates based on radio frequency (RF) fingerprints extracted from CSI. Since measured OFDM-domain CSI contains both the transmitter's RF chain impulse response (the fingerprint) and the physical RF channel, we extract RFFI features that are independent of the transmitter's location using CSI obfuscation methods inspired by [2].

The system uses:

- **CSI Features**: For each of the three DMRS symbols in a PUSCH slot, we gather CSI from all four O-RUs with four antennas each, forming a channel matrix $\mathbf{H} \in \mathbb{C}^{3276\times16}$. We stack the CSI matrices from all three DMRSs along the subcarrier dimension and normalize all columns to unit norm. We then compute the compact singular value decomposition and take the dominant left singular vector, which contains the common part across all distributed receive antennas that is mostly influenced by the transmit RF circuitry-but not by the wireless channel. The singular vector is reshaped to recover subcarrier and time dimensions, and real/imaginary parts are stacked to obtain RFFI features $\mathbf{f} \in \mathbb{R}^{3276\times3\times2}$.

- **Neural Network Architecture**: A 2D convolutional ResNet model with a softmax output layer for multi-class classification.

- **Training**: The NN is trained with categorical cross-entropy loss on one-hot ground-truth device class labels. Training is performed for at most 400 epochs with an initial learning rate of $10^{-3}$ and a batch size of 32 samples. The RMSprop optimizer is used with a learning rate scheduler that applies a step size decay of 0.2 after 10 consecutive epochs without improvement. Early stopping terminates training after 30 consecutive epochs without improvement.

- **Dataset Split**: The CAEZ-5G-DEV-CLASS (CSI Acquisition at ETH Zurich) dataset uses two testing datasets:
  - **Same-day testing**: For each UE, samples are sorted by timestamp and 12.5% from the center are used for testing (corresponding to a fraction of time when the UE was randomly moved through the lab space). The remaining 87.5% are used for training.
  - **Next-day testing**: A separate measurement campaign taken solely for testing purposes.
  
  For further information and to download the CSI dataset files (tar.zstd files), visit [https://caez.ethz.ch](https://caez.ethz.ch).

## Requirements

The code requires Python 3.x and the following packages:
- NumPy
- TensorFlow/Keras
- CuPy (for GPU-accelerated feature extraction)
- SciPy
- Matplotlib
- Seaborn
- scikit-learn
- tqdm

## Usage

### Step 1: Dataset Preparation

First, download the CAEZ-5G-DEV-CLASS dataset from [https://caez.ethz.ch](https://caez.ethz.ch). The dataset files are provided as compressed tar.zstd archives containing CSI data from the PyAerial pipeline. Uncompress the dataset and enter the filepath of the dataset location to the configuration section in `main.py`.

### Step 2: Training and Testing

Run the main training and evaluation script:

```bash
python main.py
```

This script:
- Loads CSI data using the `Load5gDataset` class
- Extracts RFFI features (CSI obfuscation features)
- Trains the ResNet classification network
- Evaluates on test sets (same-day and next-day)
- Generates confusion matrices and accuracy metrics

**Configuration**: Edit the configuration in `main.py` to specify:
- Dataset paths and device labels
- Feature extraction type (`obfuscation` or `ofdm_absolutes`)
- Training parameters (epochs, learning rate, batch size)
- GPU selection

## Implementation Notes

This simulation framework is a fork of the repository [https://github.com/gxhen/LoRa_RFFI/tree/main/Closed_set_RFFI](https://github.com/gxhen/LoRa_RFFI/tree/main/Closed_set_RFFI) that was published together with the simulation code of [1].

**Core Modification**: The main modification to the original repository is the implementation of the `Load5gDataset` class in `dataset_preparation.py`. This class:
- Loads 5G CSI from the PyAerial pipeline
- Extracts CSI obfuscation features inspired by [2]
- Handles dataset splitting for same-day and next-day testing

## File Structure

- `main.py`: Main training and evaluation script
- `dataset_preparation.py`: Dataset loading and feature extraction (`Load5gDataset` class)
- `deep_learning_models.py`: ResNet neural network architecture definitions

## Version History

- **Version 0.1**: Simulator for CAEZ-5G-DEV-CLASS experiments in [3]

## Citation

If you use this code (or parts of it), then you must cite references [1] and [3].


## References

[1] G. Shen, J. Zhang, A. Marshall, and J. Cavallaro, "Towards Scalable and Channel-Robust Radio Frequency Fingerprint Identification for LoRa," *IEEE Trans. Inf. Forensics Security*, 2022.

[2] P. Stephan, F. Euchner, and S. ten Brink, "CSI obfuscation: Single-antenna transmitters can not hide from adversarial multi-antenna radio localization systems," in *Proc. Int'l Workshop Smart Antennas (WSA)*, 2025.

[3] R. Wiesmayr, F. Zumegen, S. Taner, C. Dick, and C. Studer, "CSI-based user positioning, channel charting, and device classification with an NVIDIA 5G testbed," in *Asilomar Conf. Signals, Syst., Comput.*, Oct. 2025.