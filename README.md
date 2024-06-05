# MICN: Multi-scale Isometric Convolution Network for Time Series Forecasting (with Imputation)

This repository contains an implementation of the MICN model for time series forecasting, originally proposed in the paper "MICN: Multi-scale Local and Global Context Modeling for Long-term Series Forecasting" (ICLR 2023). This implementation has been extended to handle missing values through data imputation.

## What is MICN?

Time series forecasting is the task of predicting future values based on historical data. MICN is a deep learning model designed to excel at this task, particularly for long-term forecasting.

### Challenges in Time Series Forecasting

* **Capturing Local and Global Features:** Time series data often contains both local patterns (short-term fluctuations) and global correlations (long-term trends). Traditional models may struggle to capture both effectively.
* **Computational Efficiency:** Transformer-based models are powerful but computationally expensive for long sequences.

### How MICN Addresses These Challenges

* **Multi-scale Architecture:** MICN uses a multi-scale approach to capture patterns at different resolutions, ensuring both local and global information is captured.
* **Isometric Convolution:**  MICN employs a modified convolution operation (isometric convolution) to efficiently model global correlations without the computational burden of attention mechanisms.
* **Linear Complexity:** MICN's design achieves linear complexity with respect to the sequence length, making it more efficient than Transformer-based models for long-term forecasting.

### Imputation for Missing Data

Real-world time series data often contains missing values. This implementation incorporates data imputation techniques to fill in these gaps before feeding the data into the MICN model. This enhancement makes the model more robust and applicable to a wider range of datasets.

## Repository Contents

* `micn.py`: Implementation of the core MICN model architecture.
* `imputation.py`: Contains functions for data imputation (replace with the specific method you used).
* `train.py`: Script for training the MICN model on your dataset.
* `evaluate.py`: Script for evaluating model performance on test data.
* `config.py`: Configuration file for hyperparameters, model settings, and imputation options.


## Installation

```bash
git clone https://github.com/[your_username]/[your_repository_name].git
cd [your_repository_name]
pip install -r requirements.txt
```

## Usage

1. Prepare your data in a suitable format (CSV, NumPy, etc.).
2. Configure model and imputation settings in `config.py`.
3. Train the model: `python train.py`
4. Evaluate the model: `python evaluate.py`

## Acknowledgments

* The original MICN paper:

   ```
   @inproceedings{wang2023micn,
     title={MICN: Multi-scale Local and Global Context Modeling for Long-term Series Forecasting},
     author={Wang, Haoyu and Peng, Zhongyu and Zhang, Dezhong and Long, Mingzhu},
     booktitle={International Conference on Learning Representations},
     year={2023}
   }
   ```



