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

##  Overall Architecture
As shown in Figure 1, we decompose the time series into seasonal part and trend part by Multi-scale Hybrid Decomposition. For seasonal part, we use Seasonal Prediction block to predict. For trend part, we use simple regression to predict. 
<p align="center">
<img src=".\img\overall.png" height = "202" alt="" align=center />
<br><br>
</p>


### Seasonal Prediction block
The seasonal part contains several different patterns after Multi-scale Hybrid Decomposition. For each pattern, we use local-global module to extract local information and global correlations. 
<p align="center">
<img src=".\img\seasonal.png" height = "250" alt="" align=center />
<br><br>
</p>


### Local-Global module
We use downsampling convolution to extract local features and isometric convolution to capture global correlations. 
<p align="center">
<img src=".\img\local-global.png" height = "150" alt="" align=center />
<br><br>
</p>


### 
For a deeper understanding of the MICN model's architecture and the implementation details, refer to the **`MICN.py`** file within the **`models`** directory. This file provides detailed explanations for each building block of the model, including the multi-scale structure, isometric convolutions, and other core elements.




### Imputation for Missing Data

Real-world time series data often contains missing values. This implementation incorporates data imputation techniques to fill in these gaps before feeding the data into the MICN model. This enhancement makes the model more robust and applicable to a wider range of datasets.

## Repository Contents

* **`data`:** Contains the raw time series data used for training and evaluation.
* **`data_provider`:** Handles data loading, preprocessing, and batching for training and evaluation.
* **`exp`:**  Manages experiment sessions, including configuration, logging, and checkpointing.
* **`models`:** Contains the implementation of the MICN model architecture.
* **`run.py`:** Main script for training and evaluating the model.


### 1. Installation

```bash
git clone https://github.com/maniaa1822/NN-MICN.git
cd NN-MICN
pip install -r requirements.txt
```

### 2. Configure the Experiment

* Modify the configuration parameters that you can find in the `run.py` file.
* The parameters in the first example command are already set for the long term forecasting task.
* The parameters in the second example command are already set for the imputation task.

### 3. Run the Code

* Execute the following command in your terminal:

```bash
python3 -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_96 \
  --model MICN \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --des 'Long Term Forecasting exp' \
```

```bash
python3 -u run.py \
  --task_name imputation \
  --is_training 1 \
  --root_path ./data \
  --data_path ETTh1.csv \
  --model_id ETTh1_mask_0.125 \
  --mask_rate 0.125 \
  --model MICN \
  --data ETTh1 \
  --features MS \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 0 \
  --batch_size 16 \
  --d_model 64 \
  --d_ff 64 \
  --des 'Imputation exp' \
```


## Experiment Results

### Quantitative Results

The following table summarizes the Mean Squared Error (MSE) and Mean Absolute Error (MAE) for both the forecasting and imputation tasks:

| Task       | MSE                | MAE                |
|------------|--------------------|--------------------|
| Forecasting | 0.0643 | 0.1890 |
| Imputation  | 0.0748 | 0.1946 |

### Qualitative Results

<table align="center">
  <tr>
    <td><img src="./img/forecast.png" height="250" alt="Forecast"/></td>
    <td><img src="./img/imputation.png" height="250" alt="Imputation"/></td>
  </tr>
</table>


* **Left:** Comparison of ground truth values against the model's forecasted values.
* **Right:** Illustration of the original time series data with missing values, alongside the imputed data generated by the model.

### Loss Curve

<table align="center">
  <tr>
    <td><img src="./img/loss_train_forecast.png" height="150" alt="Training"/></td>
    <td><img src="./img/loss_vali_forecast.png" height="150" alt="Validation"/></td>
  </tr>
</table>

<table align="center">
  <tr>
    <td><img src="./img/loss_train_imputation.png" height="150" alt="Training"/></td>
    <td><img src="./img/loss_vali_imputation.png" height="150" alt="Validation"/></td>
  </tr>
</table>

The plots above shows the training/validation loss over epochs for the forecasting and imputation task. The decreasing trend indicates successful model training and convergence. Furthermore, we can obsverve the application of early stopping that successful stops the training process in order to reduce overfitting.



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



